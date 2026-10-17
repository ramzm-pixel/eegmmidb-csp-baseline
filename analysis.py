"""
CSP + LDA analysis on EEGMMIDB epochs built by epoching.py.

Evaluation arms (per subject, per window/condition/task). k = labelled calibration trials per class.

Within-subject and zero-shot arms (k = 0):
  within_loro          within-subject, leave-one-run-out
  loso_raw             leave-one-subject-out, pooled CSP+LDA, no normalisation
  loso_ea              as above after per-subject Euclidean Alignment (label-free)
  loso_ea_center       loso_ea plus per-subject feature centering (label-free, uses all the target's unlabelled trials)

Calibration arms (k > 0). All are scored on the same held-out target trials, which are the trials not used for
calibration, and are averaged over N_REPS random draws of the calibration trials:
  calib_scratch        CSP + LDA fitted from scratch on the k trials per class (no other subjects used)
  calib_newlda         pooled CSP filters, but a brand-new LDA fitted on the k trials per class (the pooled LDA is
                       thrown away). Kept to show why discarding the pooled classifier is a poor way to calibrate.
  calib_adapt_center   pooled CSP + pooled LDA; the target's features are centred using only the 2k calibration trials
  calib_adapt_means    as calib_adapt_center, and the pooled LDA's class means are also pulled toward the target's own
                       class means (a weighted average, the pooled means counting as PRIOR_WEIGHT trials per class)
Reference arms, scored on the same held-out trials as the calibration arms:
  calib_ref_ea         pooled model, no centering          (zero-shot, same test trials)
  calib_ref_ea_center  pooled model, centred with all of the target's unlabelled trials (zero-shot, same test trials)

Output: long-format CSV with one accuracy per (window, condition, task, subject, method, k).
"""
import argparse
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from common import BAD_SUBJECTS, CONDITIONS, DEFAULT_DATA_DIR, TASKS

N_COMP, CALIB_KS, N_REPS, SEED = 6, (2, 5, 10), 10, 97
PRIOR_WEIGHT = 10  # how many "virtual trials per class" the pooled class means count for in calib_adapt_means


# ---- CSP from per-trial sufficient statistics ----
def trial_stats(X):
    """X: (trials, channels, samples) -> per-trial scatter matrices and sum over trials of squared power**2."""
    S = np.einsum("ept,eqt->epq", X, X, optimize=True)
    power = np.einsum("ept,ept->et", X, X, optimize=True)
    return S, (power ** 2).sum(1)


def ledoit_wolf(S, q4, n):
    """Ledoit-Wolf shrunk covariance from a scatter matrix S (sum of x x^T over n samples)."""
    p = S.shape[0]
    emp = S / n
    mu = np.trace(emp) / p
    delta_ = np.sum(S ** 2) / n ** 2
    beta_ = (q4 / n - delta_) / (p * n)
    delta = (delta_ - 2 * mu * np.trace(emp) + p * mu ** 2) / p
    beta = min(beta_, delta)
    shrink = 0.0 if beta == 0 else beta / delta
    return (1 - shrink) * emp + shrink * mu * np.eye(p)


def fit_csp(S0, q0, n0, S1, q1, n1, n_comp=N_COMP, tol=1e-8):
    lam, U = np.linalg.eigh(S0 + S1)
    P = U[:, lam > lam.max() * tol]  # drops directions removed by ICA
    C0, C1 = ledoit_wolf(P.T @ S0 @ P, q0, n0), ledoit_wolf(P.T @ S1 @ P, q1, n1)
    ev, V = eigh(C0, C0 + C1)
    idx = np.argsort(np.abs(ev - 0.5))[::-1][:n_comp]
    return V[:, idx].T @ P.T


def features(W, S, n_samples):
    power = np.einsum("kp,npq,kq->nk", W, S, W, optimize=True) / n_samples
    return np.log(np.clip(power, 1e-12, None))


def class_stats(S, q4, y, n_samples, idx=None):
    idx = np.arange(len(y)) if idx is None else idx
    out = []
    for c in (0, 1):
        m = idx[y[idx] == c]
        out += [S[m].sum(0), float(q4[m].sum()), len(m) * n_samples]
    return out


def euclidean_align(X):
    R = np.einsum("ept,eqt->pq", X, X, optimize=True) / (X.shape[0] * X.shape[2])
    lam, U = np.linalg.eigh(R)
    inv = np.where(lam > lam.max() * 1e-6, 1 / np.sqrt(np.clip(lam, 1e-30, None)), 0.0)
    return np.einsum("pq,eqt->ept", (U * inv) @ U.T, X)


def make_subject(X, y, run):
    """X in volts, shape (trials, channels, samples)."""
    X = X * 1e6
    d = dict(y=y, run=run, T=X.shape[-1])
    d["S"], d["q"] = trial_stats(X)
    d["Se"], d["qe"] = trial_stats(euclidean_align(X))
    return d


def load_subject(path, task):
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
    return make_subject(e.get_data(copy=True), y, run)


def lda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


# ---- evaluation arms ----
def within_loro(d):
    accs = []
    for r in np.unique(d["run"]):
        te, tr = np.where(d["run"] == r)[0], np.where(d["run"] != r)[0]
        if len(np.unique(d["y"][tr])) < 2 or len(np.unique(d["y"][te])) < 2:
            continue
        W = fit_csp(*class_stats(d["S"], d["q"], d["y"], d["T"], tr))
        F = features(W, d["S"], d["T"])
        accs.append(lda().fit(F[tr], d["y"][tr]).score(F[te], d["y"][te]))
    return float(np.mean(accs)) if accs else np.nan


def pooled_filters(subs, target, aligned):
    S_key, q_key = ("Se", "qe") if aligned else ("S", "q")
    parts = [class_stats(d[S_key], d[q_key], d["y"], d["T"]) for s, d in subs.items() if s != target]
    return fit_csp(*[sum(p[i] for p in parts) for i in range(6)]), S_key


def predict_with_blended_means(pooled, F_test, class_means_target, k, prior_weight=PRIOR_WEIGHT):
    """
    Classify with the pooled LDA's covariance but class means that are a weighted average of the pooled means
    (weight prior_weight) and the target's own calibration means (weight k). Equal class priors.
    `pooled` is a fitted LinearDiscriminantAnalysis (lsqr) on classes 0 and 1.
    """
    means = (prior_weight * pooled.means_ + k * class_means_target) / (prior_weight + k)
    coef = np.linalg.solve(pooled.covariance_, means.T).T
    intercept = -0.5 * np.sum(coef * means, axis=1)
    return np.argmax(F_test @ coef.T + intercept, axis=1)


def accuracy_of(pred, y_true):
    return float(np.mean(pred == y_true))


def cross_subject_arms(subs, target, rng, calibration_only=False):
    """
    All cross-subject arms for one held-out subject. Returns a list of (method, k, accuracy).
    calibration_only skips the raw-LOSO pass and omits the k = 0 rows (loso_*), which analysis already has.
    """
    t, y = subs[target], subs[target]["y"]
    others = [s for s in subs if s != target]
    y_src = np.concatenate([subs[s]["y"] for s in others])
    out = []

    if not calibration_only:
        W_raw, key_raw = pooled_filters(subs, target, aligned=False)
        F_raw = {s: features(W_raw, subs[s][key_raw], subs[s]["T"]) for s in subs}
        F_raw_src = np.concatenate([F_raw[s] for s in others])
        out.append(("loso_raw", 0, lda().fit(F_raw_src, y_src).score(F_raw[target], y)))

    W, S_key = pooled_filters(subs, target, aligned=True)
    F = {s: features(W, subs[s][S_key], subs[s]["T"]) for s in subs}
    F_target = F[target]
    pooled_plain = lda().fit(np.concatenate([F[s] for s in others]), y_src)
    pooled_centered = lda().fit(np.concatenate([F[s] - F[s].mean(0) for s in others]), y_src)
    F_target_centered_all = F_target - F_target.mean(0)
    if not calibration_only:
        out.append(("loso_ea", 0, pooled_plain.score(F_target, y)))
        out.append(("loso_ea_center", 0, pooled_centered.score(F_target_centered_all, y)))

    for k in CALIB_KS:
        acc = {name: [] for name in ("calib_scratch", "calib_newlda", "calib_adapt_center", "calib_adapt_means",
                                     "calib_ref_ea", "calib_ref_ea_center")}
        for _ in range(N_REPS):
            cal = np.concatenate([rng.choice(np.where(y == c)[0], k, replace=False) for c in (0, 1)])
            test = np.setdiff1d(np.arange(len(y)), cal)
            if len(np.unique(y[test])) < 2:
                continue
            y_test = y[test]

            # references: pooled model with no labelled calibration, scored on the same test trials
            acc["calib_ref_ea"].append(accuracy_of(pooled_plain.predict(F_target[test]), y_test))
            acc["calib_ref_ea_center"].append(accuracy_of(pooled_centered.predict(F_target_centered_all[test]), y_test))

            # from scratch: new CSP filters and a new LDA from the k trials only
            W_s = fit_csp(*class_stats(t["S"], t["q"], y, t["T"], cal))
            F_s = features(W_s, t["S"], t["T"])
            acc["calib_scratch"].append(lda().fit(F_s[cal], y[cal]).score(F_s[test], y_test))

            # pooled filters, new LDA from the k trials only
            acc["calib_newlda"].append(lda().fit(F_target[cal], y[cal]).score(F_target[test], y_test))

            # adapt the pooled model: centre with the calibration trials, optionally also blend the class means
            F_c = F_target - F_target[cal].mean(0)
            acc["calib_adapt_center"].append(accuracy_of(pooled_centered.predict(F_c[test]), y_test))
            means_cal = np.stack([F_c[cal][y[cal] == c].mean(0) for c in (0, 1)])
            acc["calib_adapt_means"].append(
                accuracy_of(predict_with_blended_means(pooled_centered, F_c[test], means_cal, k), y_test))
        out += [(name, k, float(np.mean(v))) for name, v in acc.items() if v]
    return out


def subject_rng(subject):
    """A random generator that depends only on the seed and the subject ID, not on the order subjects are processed."""
    return np.random.default_rng([SEED, zlib.crc32(subject.encode())])


def evaluate(subs, window, condition, task, calibration_only=False):
    if len(subs) < 5:
        return []
    rows = []
    for sid, d in subs.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = [] if calibration_only else [("within_loro", 0, within_loro(d))]
            results += cross_subject_arms(subs, sid, subject_rng(sid), calibration_only)
        rows += [dict(window=window, condition=condition, task=task, subject=sid, method=m, k=k,
                      acc=a, n_trials=len(d["y"])) for m, k, a in results]
    return rows


def run_cell(epochs_dir, window, condition, task, subjects, n_jobs, calibration_only=False):
    paths = [epochs_dir / window / f"{s}_{condition}-epo.fif" for s in subjects]
    loaded = Parallel(n_jobs=n_jobs)(delayed(load_subject)(p, task) for p in paths)
    subs = {s: d for s, d in zip(subjects, loaded) if d is not None}
    return evaluate(subs, window, condition, task, calibration_only)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--epochs-dir", type=Path, default=None, help="default: <data-dir>/epochs")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: results/results_long.csv, or results/results_calibration.csv with --calibration-only")
    ap.add_argument("--windows", nargs="+", default=["0.0 - 4.0"])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--calibration-only", action="store_true",
                    help="compute only the calibration and reference arms (fast); write them to a separate file")
    ap.add_argument("--n-jobs", type=int, default=-2)
    a = ap.parse_args()

    epochs_dir = a.epochs_dir or a.data_dir / "epochs"
    out_path = a.out or Path("results") / ("results_calibration.csv" if a.calibration_only else "results_long.csv")
    subjects = sorted(p.name for p in a.data_dir.iterdir()
                      if p.is_dir() and p.name.startswith("S") and p.name not in BAD_SUBJECTS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(out_path).to_dict("records") if out_path.exists() else []
    done = {(r["window"], r["condition"], r["task"]) for r in rows}
    for w in a.windows:
        for c in a.conditions:
            for t in a.tasks:
                if (w, c, t) in done:
                    continue
                rows += run_cell(epochs_dir, w, c, t, subjects, a.n_jobs, a.calibration_only)
                pd.DataFrame(rows).to_csv(out_path, index=False)
                print("done", w, c, t, flush=True)


if __name__ == "__main__":
    main()
