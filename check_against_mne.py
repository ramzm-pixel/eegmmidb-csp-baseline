"""
Cross-check analysis.py against MNE's own CSP + scikit-learn LDA on real epoch files.

analysis.py computes CSP from per-trial summary statistics instead of calling mne.decoding.CSP. This script runs both
implementations on the same trials and compares the resulting accuracies:

  within-subject   leave-one-run-out accuracy, per subject         (analysis.within_loro  vs  MNE pipeline)
  zero-shot        leave-one-subject-out raw accuracy, per subject (analysis loso_raw      vs  MNE pipeline)
  alignment        after Euclidean Alignment the average covariance of every subject should be the identity
                   (on the directions that carry signal)

Use a small pool for the zero-shot check (default 12 subjects) so the MNE side stays fast. Small differences are
expected: MNE orders and selects CSP components by mutual information and handles rank deficiency its own way, so a
trial near the decision boundary can flip. Differences that are large or one-sided are not expected.

    python check_against_mne.py --data-dir data/eegmmidb --window "0.0 - 4.0" --condition motor_ica --task real_lr
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import analysis
from common import BAD_SUBJECTS, CONDITIONS, DEFAULT_DATA_DIR, TASKS

MEAN_TOLERANCE = 0.01  # mean absolute difference in accuracy that counts as agreement


def mne_pipeline():
    """The textbook pipeline with the same settings analysis.py claims to reproduce."""
    from sklearn.pipeline import Pipeline
    return Pipeline([("csp", CSP(n_components=analysis.N_COMP, reg="ledoit_wolf", log=True, norm_trace=False)),
                     ("lda", analysis.lda())])


def load_trials(path, task):
    """Raw (volts) trials, labels and run numbers, selected exactly as analysis.load_subject does."""
    import mne
    mne.set_log_level("ERROR")
    path = Path(path)
    if not path.exists():
        return None
    epochs = mne.read_epochs(path, preload=True, verbose=False)
    a, b = TASKS[task]
    if a not in epochs.event_id or b not in epochs.event_id:
        return None
    e = epochs[[a, b]]
    y = (e.events[:, -1] == e.event_id[a]).astype(int)
    run = e.metadata["run"].to_numpy()
    if min(y.sum(), (1 - y).sum()) < 6 or len(np.unique(run)) < 2:
        return None
    return e.get_data(copy=True), y, run


def mne_within_loro(X, y, run):
    accs = []
    for r in np.unique(run):
        te, tr = run == r, run != r
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            accs.append(mne_pipeline().fit(X[tr], y[tr]).score(X[te], y[te]))
    return float(np.mean(accs)) if accs else np.nan


def mne_loso(trials, target):
    others = [s for s in trials if s != target]
    X_train = np.concatenate([trials[s][0] for s in others])
    y_train = np.concatenate([trials[s][1] for s in others])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mne_pipeline().fit(X_train, y_train).score(trials[target][0], trials[target][1])


def alignment_report(trials):
    rows = []
    for s, (X, _, _) in trials.items():
        Xa = analysis.euclidean_align(X * 1e6)
        R = np.einsum("ept,eqt->pq", Xa, Xa) / (Xa.shape[0] * Xa.shape[2])
        lam = np.linalg.eigvalsh(R)
        kept = lam[lam > 0.5]
        rows.append(dict(subject=s, channels=len(lam), kept_directions=len(kept),
                         max_deviation_from_1=float(np.abs(kept - 1).max()) if len(kept) else np.nan))
    return pd.DataFrame(rows)


def compare(name, a, b):
    d = pd.DataFrame({"analysis_py": a, "mne": b}).dropna()
    d["abs_diff"] = (d.analysis_py - d.mne).abs()
    print(f"\n== {name} ({len(d)} subjects)")
    print(f"mean accuracy   analysis.py {d.analysis_py.mean():.4f}   MNE {d.mne.mean():.4f}")
    print(f"mean |difference| {d.abs_diff.mean():.4f}   max {d.abs_diff.max():.4f}   "
          f"identical for {(d.abs_diff < 1e-9).mean():.0%} of subjects   correlation {d.analysis_py.corr(d.mne):.3f}")
    ok = d.abs_diff.mean() <= MEAN_TOLERANCE
    print("AGREE" if ok else f"DISAGREE (mean |difference| above {MEAN_TOLERANCE}) - do not publish before investigating")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--epochs-dir", type=Path, default=None, help="default: <data-dir>/epochs")
    ap.add_argument("--window", default="0.0 - 4.0")
    ap.add_argument("--condition", default="motor_ica", choices=CONDITIONS)
    ap.add_argument("--task", default="real_lr", choices=list(TASKS))
    ap.add_argument("--n-subjects", type=int, default=12, help="subjects used (the LOSO pool is these subjects only)")
    a = ap.parse_args()

    epochs_dir = a.epochs_dir or a.data_dir / "epochs"
    names = sorted(p.name for p in a.data_dir.iterdir()
                   if p.is_dir() and p.name.startswith("S") and p.name not in BAD_SUBJECTS)
    trials = {}
    for s in names:
        t = load_trials(epochs_dir / a.window / f"{s}_{a.condition}-epo.fif", a.task)
        if t is not None:
            trials[s] = t
        if len(trials) == a.n_subjects:
            break
    if len(trials) < 5:
        raise SystemExit(f"Only {len(trials)} usable subjects found under {epochs_dir / a.window}; need at least 5.")
    print(f"window {a.window} | {a.condition} | {a.task} | subjects: {', '.join(trials)}")

    subs = {s: analysis.make_subject(X, y, run) for s, (X, y, run) in trials.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = pd.DataFrame(analysis.evaluate(subs, a.window, a.condition, a.task))
    ours = ours[ours.k == 0].pivot(index="subject", columns="method", values="acc")

    ok = compare("within-subject, leave-one-run-out", ours["within_loro"],
                 pd.Series({s: mne_within_loro(*t) for s, t in trials.items()}))
    ok &= compare("zero-shot, raw (leave-one-subject-out)", ours["loso_raw"],
                  pd.Series({s: mne_loso(trials, s) for s in trials}))

    print("\n== Euclidean Alignment: average covariance of each aligned subject")
    rep = alignment_report(trials)
    print(rep.to_string(index=False))
    good = (rep.max_deviation_from_1 < 1e-6).all()
    print("AGREE (identity on every kept direction)" if good else "CHECK: aligned covariance is not the identity")
    print("\nOVERALL:", "PASS" if ok and good else "FAIL")


if __name__ == "__main__":
    main()
