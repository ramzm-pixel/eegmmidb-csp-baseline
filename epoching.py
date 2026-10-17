"""
Build epochs for the 2x2 preprocessing design (channel set x ICA cleaning) and every analysis window.

Conditions (see common.CONDITIONS):
    motor_raw   21 motor channels, no ICA        hybrid_raw   42 channels (motor + frontal/ocular), no ICA
    motor_ica   21 motor channels, ICA cleaned   hybrid_ica   42 channels, ICA cleaned
ICA is fitted once per subject on all 64 channels and applied before channel selection.
Each epoch carries its recording run number in epochs.metadata["run"] for leave-one-run-out validation.
"""
import argparse
import re
import traceback
from pathlib import Path
from typing import cast

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import BAD_SUBJECTS, DEFAULT_DATA_DIR, MOTOR_CHANNELS, OCULAR_ARTIFACT_CHANNELS

WINDOW_LENGTH, STEP_SIZE, T_START, T_END = 1.5, 0.25, 0.0, 5.5
ICA_DECIM = 2

RUN_RE = re.compile(r"R(\d{2})", re.IGNORECASE)
RUN_MAPPINGS = {
    **{r: {"T0": "rest_fist_lr_real", "T1": "left_fist_real", "T2": "right_fist_real"} for r in (3, 7, 11)},
    **{r: {"T0": "rest_fist_lr_imagined", "T1": "left_fist_imagined", "T2": "right_fist_imagined"} for r in (4, 8, 12)},
    **{r: {"T0": "rest_fist_feet_real", "T1": "both_fists_real", "T2": "both_feet_real"} for r in (5, 9, 13)},
    **{r: {"T0": "rest_fist_feet_imagined", "T1": "both_fists_imagined", "T2": "both_feet_imagined"} for r in (6, 10, 14)},
}


def sliding_windows() -> list[tuple[float, float]]:
    windows, t = [], T_START
    while t + WINDOW_LENGTH <= T_END + 1e-3:
        windows.append((round(t, 2), round(t + WINDOW_LENGTH, 2)))
        t += STEP_SIZE
    windows.append((0.0, 4.0))
    return windows


def window_name(w: tuple[float, float]) -> str:
    return f"{w[0]} - {w[1]}"


def run_number(path: Path) -> int:
    match = RUN_RE.search(path.stem)
    if not match:
        raise ValueError(f"Could not parse run number from {path.name}")
    return int(match.group(1))


def load_subject_raw(data_dir: Path, subject: str):
    """Concatenate task runs 3-14. Returns (raw, run numbers, start sample of each run)."""
    files = sorted((data_dir / subject).glob("*.edf"))
    if not files:
        raise FileNotFoundError(f"No EDF files found for {subject} under {data_dir}")
    raws, runs = [], []
    for f in files:
        run = run_number(f)
        if run in (1, 2):  # baseline recordings
            continue
        raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
        mapping = RUN_MAPPINGS[run]
        raw.annotations.description = np.array([mapping.get(d, d) for d in raw.annotations.description])
        raws.append(raw)
        runs.append(run)
    if not raws:
        raise RuntimeError(f"No task runs (R03-R14) found for {subject}")
    run_starts = np.cumsum([0] + [r.n_times for r in raws[:-1]])
    combined = cast(mne.io.BaseRaw, mne.concatenate_raws(raws))
    combined.rename_channels({ch: ch.rstrip(".").upper() for ch in combined.ch_names})
    combined.set_montage(mne.channels.make_standard_montage("standard_1005"), match_case=False, on_missing="warn")
    return combined, np.array(runs), run_starts


def run_ica(raw: mne.io.BaseRaw) -> tuple[mne.io.BaseRaw, bool]:
    """Remove EOG and muscle components. Returns (cleaned copy, whether the EOG-only fallback was used)."""
    ica_input = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    ica = mne.preprocessing.ICA(n_components=0.99, method="fastica", random_state=97)
    ica.fit(ica_input, decim=ICA_DECIM, verbose=False)

    frontal = [ch for ch in ("FP1", "FP2", "FPZ", "AF7", "AF8") if ch in ica_input.ch_names]
    eog_idx: list[int] = []
    if frontal:
        try:
            eog_idx, _ = ica.find_bads_eog(ica_input, ch_name=frontal, verbose=False)
        except RuntimeError:
            pass
    try:
        muscle_idx, _ = ica.find_bads_muscle(ica_input, threshold=1.5)
    except Exception:
        muscle_idx = []

    bad = sorted(set(eog_idx) | set(muscle_idx))
    fallback = bool(ica.n_components_ and len(bad) > 0.3 * ica.n_components_)
    if fallback:  # implausibly many muscle components: keep only the EOG ones
        bad = list(eog_idx)
    ica.exclude = bad
    cleaned = raw.copy()
    ica.apply(cleaned, verbose=False)
    return cleaned, fallback


def preprocess(raw: mne.io.BaseRaw, ocular: bool) -> mne.io.BaseRaw:
    channels = MOTOR_CHANNELS + (OCULAR_ARTIFACT_CHANNELS if ocular else [])
    out = raw.copy().pick(channels)
    out.notch_filter(60, verbose=False)
    out.filter(l_freq=8, h_freq=30, verbose=False)
    return out


def epoch_and_save(data, tag, runs, run_starts, windows, epochs_dir, subject) -> None:
    ocular = tag.startswith("hybrid")
    filtered = preprocess(data, ocular)
    events, event_id = mne.events_from_annotations(filtered, verbose=False)
    event_id = {k: v for k, v in event_id.items() if not k.startswith("rest")}
    events = events[np.isin(events[:, 2], list(event_id.values()))]
    run_idx = np.searchsorted(run_starts, events[:, 0] - filtered.first_samp, side="right") - 1
    metadata = pd.DataFrame({"run": runs[run_idx]})

    base = mne.Epochs(filtered, events, event_id, tmin=T_START, tmax=T_END, baseline=None,
                      preload=True, metadata=metadata, verbose=False)
    for w in windows:
        out = epochs_dir / window_name(w) / f"{subject}_{tag}-epo.fif"
        if not out.exists():
            base.copy().crop(tmin=w[0], tmax=w[1]).save(out, overwrite=True, verbose=False)


def process_subject(subject: str, data_dir: Path, epochs_dir: Path, windows) -> tuple[str, int, str | None]:
    try:
        from common import CONDITIONS
        for w in windows:
            (epochs_dir / window_name(w)).mkdir(parents=True, exist_ok=True)
        missing = [c for c in CONDITIONS
                   if not all((epochs_dir / window_name(w) / f"{subject}_{c}-epo.fif").exists() for w in windows)]
        if not missing:
            return subject, 0, None
        raw, runs, run_starts = load_subject_raw(data_dir, subject)
        fallback = 0
        cleaned = None
        for tag in missing:
            if tag.endswith("_ica"):
                if cleaned is None:
                    cleaned, fb = run_ica(raw)
                    fallback = int(fb)
                data = cleaned
            else:
                data = raw
            epoch_and_save(data, tag, runs, run_starts, windows, epochs_dir, subject)
        return subject, fallback, None
    except Exception as e:
        traceback.print_exc()
        return subject, 0, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="EEGMMIDB root (contains S001, S002, ...)")
    ap.add_argument("--epochs-dir", type=Path, default=None, help="default: <data-dir>/epochs")
    ap.add_argument("--n-jobs", type=int, default=-2, help="parallel subjects; each worker needs ~1-2 GB RAM")
    a = ap.parse_args()
    epochs_dir = a.epochs_dir or a.data_dir / "epochs"
    windows = sliding_windows()
    subjects = sorted(p.name for p in a.data_dir.iterdir()
                      if p.is_dir() and p.name.startswith("S") and p.name not in BAD_SUBJECTS)
    print(f"{len(subjects)} subjects, {len(windows)} windows, excluded: {sorted(BAD_SUBJECTS)}")
    results = Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(process_subject)(s, a.data_dir, epochs_dir, windows) for s in subjects)
    print(f"ICA EOG-only fallback used for {sum(r[1] for r in results)} subjects")
    failed = [(s, e) for s, _, e in results if e]
    for s, e in failed:
        print(f"[{s}] FAILED: {e}")
    if failed:
        print("Re-run to retry; finished files are skipped.")


if __name__ == "__main__":
    main()
