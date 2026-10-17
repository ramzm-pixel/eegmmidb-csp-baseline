# CSP+LDA motor-imagery baselines on EEGMMIDB

**Read the report: [REPORT.md](REPORT.md)** (results, figures, limitations). Statistics for all 80 pre-specified tests are in `results/stats_tests.csv`.

A controlled analysis of a standard CSP + LDA decoder on the PhysioNet EEG Motor Movement/Imagery dataset (105 subjects after excluding S088, S089, S092, S100).

## Questions
1. Preprocessing: effect of ICA cleaning and of channel set (21 motor vs 42 motor + frontal/ocular), as a 2x2 design.
2. Cross-subject transfer: leave-one-subject-out (LOSO) with and without Euclidean Alignment, and a calibration curve comparing training from scratch on k labelled trials, retraining only the classifier, and adapting the pooled model (centring and blending class means) with the same k trials.
3. Timing: which 1.5 s window after the cue is best (equal-length windows only; the 4 s task window is reported separately because window length is confounded with covariance-estimation quality).
4. Real vs imagined movement, and left/right fist vs fists/feet.

Features are 8-30 Hz log band power (ERD/ERS).

## Validation
Within-subject accuracy uses leave-one-run-out cross-validation. Cross-subject arms are evaluated on every trial of the held-out subject, except the calibration arms, which are scored on the trials not used for calibration. All contrasts are paired across subjects (Wilcoxon), pre-specified in `stats.py`, and corrected with Holm over the whole family.

## Usage
```
pip install -r requirements.txt   # or requirements-lock.txt for the tested versions
python epoching.py --data-dir data/eegmmidb          # EDF files in data/eegmmidb/S001/...
python analysis.py --data-dir data/eegmmidb          # primary window 0.0 - 4.0
python analysis.py --data-dir data/eegmmidb --windows "0.0 - 1.5" "0.25 - 1.75" "0.5 - 2.0" ...  # timing family
python stats.py --results results/results_long.csv results/results_calibration.csv
python make_figures.py --results results/results_long.csv results/results_calibration.csv   # figures/ and results/tables.md
python check_against_mne.py --data-dir data/eegmmidb   # cross-check against MNE's CSP on real epochs
pytest
```
Both scripts are resumable. Epoching needs roughly 1-2 GB of RAM per parallel worker (`--n-jobs`).

## Data and results files
The dataset is not included; download EEGMMIDB from PhysioNet (https://physionet.org/content/eegmmidb/1.0.0/) into `data/eegmmidb/`. `results/results_long.csv.gz` holds every per-subject accuracy (decompress it to `results/results_long.csv` to resume or re-run `stats.py`; `make_figures.py` can read the `.gz` directly). `results/stats_tests.csv` holds the 80 tests.

## Files
`common.py` constants | `epoching.py` epochs for 4 conditions x all windows, with run metadata | `analysis.py` decoding arms | `stats.py` corrected statistics | `make_figures.py` figures and tables for the report | `check_against_mne.py` agreement check with `mne.decoding.CSP` | `test_analysis.py` synthetic-data tests (they check that the code is correct, not that results are good; they need no dataset)
