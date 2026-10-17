"""
Pre-specified paired contrasts on results_long.csv, corrected for multiple comparisons (Holm) over the whole family.
Primary setting: window 0.0 - 4.0, condition motor_ica unless a contrast says otherwise.
Also reports: share of subjects above the binomial chance threshold, and a Friedman test over equal-length (1.5 s)
windows for the timing question. Effect size: matched-pairs rank-biserial correlation.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binom
from statsmodels.stats.multitest import multipletests

PRIMARY_WINDOW, PRIMARY_CONDITION = "0.0 - 4.0", "motor_ica"
TASKS = ["real_lr", "imagined_lr", "real_feet", "imagined_feet"]
CALIB_KS = (2, 5, 10)


def accuracy(df, task, method="within_loro", k=0, condition=PRIMARY_CONDITION, window=PRIMARY_WINDOW):
    s = df[(df.window == window) & (df.condition == condition) & (df.task == task)
           & (df.method == method) & (df.k == k)]
    return s.set_index("subject").acc


def rank_biserial(diff):
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    r = stats.rankdata(np.abs(diff))
    return float((r[diff > 0].sum() - r[diff < 0].sum()) / r.sum())


def contrast(tests, family, label, a, b):
    pair = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if len(pair) < 10:
        return
    diff = (pair.a - pair.b).to_numpy()
    p = stats.wilcoxon(pair.a, pair.b, zero_method="pratt").pvalue if np.any(diff != 0) else 1.0
    tests.append(dict(family=family, contrast=label, n=len(pair), mean_a=pair.a.mean(), mean_b=pair.b.mean(),
                      diff=diff.mean(), rank_biserial=rank_biserial(diff), p=p))


def build_tests(df):
    t = []
    for task in TASKS:
        # preprocessing: ICA effect and channel-set effect, each with the other factor held fixed
        for ch in ("motor", "hybrid"):
            contrast(t, "preprocessing", f"{task}: ICA vs no ICA ({ch} channels)",
                     accuracy(df, task, condition=f"{ch}_ica"), accuracy(df, task, condition=f"{ch}_raw"))
        for ica in ("raw", "ica"):
            contrast(t, "preprocessing", f"{task}: 21 motor vs 42 channels ({ica})",
                     accuracy(df, task, condition=f"motor_{ica}"), accuracy(df, task, condition=f"hybrid_{ica}"))
        # cross-subject transfer
        contrast(t, "cross-subject", f"{task}: LOSO aligned vs raw",
                 accuracy(df, task, "loso_ea"), accuracy(df, task, "loso_raw"))
        contrast(t, "cross-subject", f"{task}: LOSO aligned+centered vs aligned",
                 accuracy(df, task, "loso_ea_center"), accuracy(df, task, "loso_ea"))
        contrast(t, "cross-subject", f"{task}: within-subject vs LOSO aligned+centered",
                 accuracy(df, task, "within_loro"), accuracy(df, task, "loso_ea_center"))
        # calibration (all arms are scored on the same held-out trials for a given k)
        for k in CALIB_KS:
            a = lambda method: accuracy(df, task, method, k)
            contrast(t, "calibration", f"{task}: adapted pooled model vs scratch, k={k}/class",
                     a("calib_adapt_means"), a("calib_scratch"))
            contrast(t, "calibration", f"{task}: adapted pooled model vs zero-shot centred, k={k}/class",
                     a("calib_adapt_means"), a("calib_ref_ea_center"))
            contrast(t, "calibration", f"{task}: adapt means vs centre only, k={k}/class",
                     a("calib_adapt_means"), a("calib_adapt_center"))
            contrast(t, "calibration", f"{task}: new LDA only vs zero-shot centred, k={k}/class",
                     a("calib_newlda"), a("calib_ref_ea_center"))
    # task comparisons (within-subject, same subjects)
    for a, b in (("real_lr", "imagined_lr"), ("real_feet", "imagined_feet"),
                 ("real_feet", "real_lr"), ("imagined_feet", "imagined_lr")):
        contrast(t, "task", f"{a} vs {b}", accuracy(df, a), accuracy(df, b))
    out = pd.DataFrame(t)
    out["p_holm"] = multipletests(out.p, method="holm")[1]
    out["p_bh"] = multipletests(out.p, method="fdr_bh")[1]
    return out


def chance_report(df):
    def threshold(n):
        return binom.ppf(0.95, n, 0.5) / n
    d = df[(df.window == PRIMARY_WINDOW) & (df.condition == PRIMARY_CONDITION) & (df.k == 0)].copy()  # k = 0: no calibration trials
    d["above"] = d.acc > d.n_trials.map(threshold)
    return d.groupby(["task", "method"]).agg(mean_acc=("acc", "mean"), share_above_chance=("above", "mean")).round(3)


def timing_report(df):
    d = df[(df.condition == PRIMARY_CONDITION) & (df.method == "within_loro")]
    lengths = d.window.map(lambda w: float(w.split(" - ")[1]) - float(w.split(" - ")[0]))
    d = d[np.isclose(lengths, 1.5)]
    rows = []
    for task, g in d.groupby("task"):
        pv = g.pivot(index="subject", columns="window", values="acc").dropna()
        if pv.shape[1] < 3:
            continue
        chi2, p = stats.friedmanchisquare(*[pv[c] for c in pv.columns])
        rows.append(dict(task=task, windows=pv.shape[1], n=len(pv), chi2=chi2, p=p,
                         best_window=pv.mean().idxmax(), best_mean=pv.mean().max()))
    return pd.DataFrame(rows).round(4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, nargs="+", default=[Path("results") / "results_long.csv"],
                    help="one or more result files; for the same (window, condition, task, subject, method, k) "
                         "the later file wins, so list results_calibration.csv last")
    ap.add_argument("--out", type=Path, default=Path("results") / "stats_tests.csv")
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(p) for p in a.results], ignore_index=True)
    df = df.drop_duplicates(["window", "condition", "task", "subject", "method", "k"], keep="last")
    tests = build_tests(df)
    tests.to_csv(a.out, index=False)
    pd.set_option("display.width", 200, "display.max_colwidth", 70, "display.max_rows", 200)
    print(f"{len(tests)} pre-specified tests | uncorrected p<.05: {(tests.p < .05).sum()} | Holm<.05: {(tests.p_holm < .05).sum()}")
    print(tests[tests.p_holm < .05][["contrast", "diff", "rank_biserial", "p_holm"]].round(4).to_string(index=False))
    print("\nChance-level report:\n", chance_report(df))
    print("\nWindow timing (equal-length 1.5 s windows):\n", timing_report(df).to_string(index=False))


if __name__ == "__main__":
    main()
