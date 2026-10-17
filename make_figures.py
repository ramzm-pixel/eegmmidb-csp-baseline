"""
Figures and summary tables for the report, built from the long-format results.

    python make_figures.py --results results/results_long.csv [results/results_calibration.csv ...]

Writes figures/*.png and results/tables.md. Later result files override earlier ones for the same
(window, condition, task, subject, method, k), exactly as in stats.py.

Intervals are mean +/- 1.96 standard errors over subjects. Because every comparison in the report is paired
across subjects, the paired tests in stats.py are the evidence; the intervals here are descriptive.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom

from stats import CALIB_KS, PRIMARY_CONDITION, PRIMARY_WINDOW, TASKS, accuracy

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdbd6"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TASK_LABEL = {"real_lr": "real L/R", "imagined_lr": "imagined L/R",
              "real_feet": "real feet", "imagined_feet": "imagined feet"}
METHOD_LABEL = {"within_loro": "within-subject (leave-one-run-out)", "loso_raw": "zero-shot, raw",
                "loso_ea": "zero-shot, aligned", "loso_ea_center": "zero-shot, aligned + centred"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 100,
})


def load(paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return df.drop_duplicates(["window", "condition", "task", "subject", "method", "k"], keep="last")


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    se = x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), 1.96 * se


def chance_threshold(n):
    return binom.ppf(0.95, n, 0.5) / n


# ---------------------------------------------------------------- figures
def fig_transfer(df, out):
    methods = ["within_loro", "loso_raw", "loso_ea", "loso_ea_center"]
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    offsets = np.linspace(-0.27, 0.27, len(methods))
    for m, c, off in zip(methods, colors, offsets):
        mu, ci = zip(*[mean_ci(accuracy(df, t, m)) for t in TASKS])
        ax.errorbar(np.arange(len(TASKS)) + off, mu, yerr=ci, fmt="o", color=c, ms=8, mec=SURFACE, mew=1.5,
                    capsize=0, lw=2, label=METHOD_LABEL[m])
    ax.axhline(0.5, color=INK2, lw=1, ls=":")
    ax.text(len(TASKS) - 0.52, 0.505, "chance", color=INK2, ha="right", va="bottom", fontsize=9)
    ax.set_xticks(range(len(TASKS)), [TASK_LABEL[t] for t in TASKS])
    ax.set_ylim(0.45, 0.85)
    ax.set_ylabel("accuracy (mean over 105 subjects, 95% interval)")
    ax.set_title("Zero-shot transfer works once subjects are aligned")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


CAL_ARMS = [("calib_scratch", "from scratch (k trials only)", BLUE, "o"),
            ("calib_newlda", "pooled filters, new LDA", ORANGE, "s"),
            ("calib_adapt_center", "adapt: centre only", AQUA, "^"),
            ("calib_adapt_means", "adapt: centre + blend means", YELLOW, "D")]


def fig_calibration(df, out):
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), sharey=True, sharex=True)
    for ax, task in zip(axes.ravel(), TASKS):
        for m, label, c, mk in CAL_ARMS:
            mu = [accuracy(df, task, m, k).mean() for k in CALIB_KS]
            ax.plot(CALIB_KS, mu, color=c, marker=mk, ms=7, mec=SURFACE, mew=1.2, lw=2.2, label=label)
        ref = [accuracy(df, task, "calib_ref_ea_center", k).mean() for k in CALIB_KS]
        ax.plot(CALIB_KS, ref, color=INK, lw=1.6, ls="--", label="zero-shot, aligned + centred (no labels)")
        ax.axhline(accuracy(df, task, "within_loro").mean(), color=INK2, lw=1.2, ls=":",
                   label="within-subject, leave-one-run-out")
        ax.set_title(TASK_LABEL[task])
        ax.set_xticks(CALIB_KS)
        ax.set_ylim(0.5, 0.82)
        ax.grid(axis="x", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("labelled calibration trials per class (k)")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy (mean over subjects)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("A few labelled trials do not beat zero-shot with centring", x=0.065, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    fig.savefig(out, dpi=170)
    plt.close(fig)


def window_starts(df):
    d = df[(df.condition == PRIMARY_CONDITION) & (df.method == "within_loro")].copy()
    lengths = d.window.map(lambda w: float(w.split(" - ")[1]) - float(w.split(" - ")[0]))
    d = d[np.isclose(lengths, 1.5)]
    d["start"] = d.window.map(lambda w: float(w.split(" - ")[0]))
    return d


def fig_timing(df, out):
    d = window_starts(df)
    style = {"real_lr": (BLUE, "-"), "imagined_lr": (BLUE, "--"), "real_feet": (ORANGE, "-"),
             "imagined_feet": (ORANGE, "--")}
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for t in TASKS:
        g = d[d.task == t].groupby("start").acc.mean()
        c, ls = style[t]
        ax.plot(g.index, g.values, color=c, ls=ls, lw=2.2, marker="o", ms=5, mec=SURFACE, mew=1, label=TASK_LABEL[t])
    ax.axvline(2.5, color=INK2, lw=1, ls=":")
    ax.text(2.55, 0.505, "windows starting later than 2.5 s\nextend past the 4 s task period", color=INK2,
            fontsize=8.5, va="bottom")
    ax.set_xlabel("start of the 1.5 s window after the cue (s)")
    ax.set_ylabel("within-subject accuracy (mean over subjects)")
    ax.set_ylim(0.5, 0.82)
    ax.set_xticks(np.arange(0, 4.01, 0.5))
    ax.set_title("Feet tasks peak about 1 s after the cue; left/right barely varies")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def per_subject_table(df):
    d = df[(df.window == PRIMARY_WINDOW) & (df.condition == PRIMARY_CONDITION) & (df.k == 0)
           & df.method.isin(["within_loro", "loso_ea_center"])].copy()
    d["above"] = d.acc > d.n_trials.map(chance_threshold)
    return d


def fig_subjects(df, out):
    d = per_subject_table(df)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ticks = []
    for i, t in enumerate(TASKS):
        shares = []
        for j, (m, c) in enumerate((("within_loro", BLUE), ("loso_ea_center", ORANGE))):
            g = d[(d.task == t) & (d.method == m)]
            x = i + (-0.19 if j == 0 else 0.19)
            ax.scatter(x + rng.uniform(-0.1, 0.1, len(g)), g.acc, s=14, color=c, alpha=0.55, lw=0)
            ax.hlines(g.acc.median(), x - 0.15, x + 0.15, color=INK, lw=2.4, zorder=3)
            shares.append(f"{g.above.mean():.0%}")
        ticks.append(f"{TASK_LABEL[t]}\nabove chance:\n{shares[0]} | {shares[1]}")
    ax.axhline(chance_threshold(42), color=INK2, lw=1, ls="--")
    ax.set_xticks(range(len(TASKS)), ticks)
    ax.set_ylim(0.33, 1.02)
    ax.set_xlim(-0.55, len(TASKS) - 0.45)
    ax.set_ylabel("per-subject accuracy (bar = median)")
    ax.set_title("Most of the spread is between people, not between methods")
    ax.grid(axis="x", visible=False)
    ax.scatter([], [], color=BLUE, s=30, label="within-subject (leave-one-run-out)")
    ax.scatter([], [], color=ORANGE, s=30, label="zero-shot, aligned + centred")
    ax.plot([], [], color=INK2, lw=1, ls="--", label="95% chance level at 42 trials")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def preprocessing_effects(df):
    rows = []
    for factor, color in (("ICA vs no ICA", BLUE), ("21 motor vs 42 channels", ORANGE)):
        for t in TASKS:
            for variant in ("motor", "hybrid") if factor.startswith("ICA") else ("raw", "ica"):
                if factor.startswith("ICA"):
                    a, b, tag = (accuracy(df, t, condition=f"{variant}_ica"), accuracy(df, t, condition=f"{variant}_raw"),
                                 f"{variant} channels")
                else:
                    a, b, tag = (accuracy(df, t, condition=f"motor_{variant}"),
                                 accuracy(df, t, condition=f"hybrid_{variant}"), "no ICA" if variant == "raw" else "ICA")
                diff = (a - b).dropna()
                mu, ci = mean_ci(diff)
                rows.append(dict(factor=factor, task=t, setting=tag, diff=mu, ci=ci, color=color))
    return pd.DataFrame(rows)


def fig_preprocessing(df, out):
    e = preprocessing_effects(df)
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    y = np.arange(len(e))[::-1].astype(float)
    y[len(e) // 2:] -= 1
    for yy, r in zip(y, e.itertuples()):
        ax.errorbar(100 * r.diff, yy, xerr=100 * r.ci, fmt="o", color=r.color, ms=8, mec=SURFACE, mew=1.5, lw=2)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y, [f"{TASK_LABEL[r.task]}, {r.setting}" for r in e.itertuples()])
    ax.set_xlabel("difference in accuracy (percentage points, mean of paired differences, 95% interval)")
    ax.set_xlim(-4, 6)
    ax.grid(axis="y", visible=False)
    ax.text(5.9, y[0] + 0.5, "ICA minus no ICA", color=BLUE, ha="right", fontweight="bold", fontsize=9.5)
    ax.text(5.9, y[len(e) // 2] + 0.5, "21 motor channels minus 42 channels", color=ORANGE, ha="right",
            fontweight="bold", fontsize=9.5)
    ax.set_title("ICA makes no consistent difference; fewer channels is slightly better")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- tables
def md(df, floatfmt=".3f"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:{floatfmt}}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def tables(df):
    out = {}
    transfer = pd.DataFrame({TASK_LABEL[t]: {METHOD_LABEL[m]: accuracy(df, t, m).mean() for m in METHOD_LABEL}
                             for t in TASKS})
    out["transfer"] = md(transfer.reset_index().rename(columns={"index": "method"}))

    rows = []
    for t in TASKS:
        for k in CALIB_KS:
            row = {"task": TASK_LABEL[t], "k": k}
            for m, label, _, _ in CAL_ARMS:
                row[label] = accuracy(df, t, m, k).mean()
            row["zero-shot, aligned + centred"] = accuracy(df, t, "calib_ref_ea_center", k).mean()
            rows.append(row)
    out["calibration"] = md(pd.DataFrame(rows))

    e = preprocessing_effects(df)
    e["effect (points)"] = (100 * e["diff"]).round(2)
    e["95% interval (points)"] = [f"±{100 * c:.2f}" for c in e["ci"]]
    e["task"] = e.task.map(TASK_LABEL)
    out["preprocessing"] = md(e[["factor", "task", "setting", "effect (points)", "95% interval (points)"]], ".2f")

    d = per_subject_table(df)
    s = d.groupby(["task", "method"]).agg(mean=("acc", "mean"), sd=("acc", "std"), above=("above", "mean")).reset_index()
    s["task"] = s.task.map(TASK_LABEL)
    s["method"] = s.method.map(METHOD_LABEL)
    s = s.rename(columns={"above": "share above chance"})
    s["share above chance"] = s["share above chance"].map("{:.0%}".format)
    out["subjects"] = md(s)

    t = window_starts(df)
    timing = t.groupby(["start", "task"]).acc.mean().unstack()[TASKS].rename(columns=TASK_LABEL)
    timing.insert(0, "window start (s)", timing.index)
    out["timing"] = md(timing.reset_index(drop=True))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, nargs="+", default=[Path("results") / "results_long.csv"])
    ap.add_argument("--figures", type=Path, default=Path("figures"))
    ap.add_argument("--tables", type=Path, default=Path("results") / "tables.md")
    a = ap.parse_args()
    df = load(a.results)
    a.figures.mkdir(parents=True, exist_ok=True)
    a.tables.parent.mkdir(parents=True, exist_ok=True)
    fig_transfer(df, a.figures / "transfer.png")
    fig_calibration(df, a.figures / "calibration.png")
    fig_timing(df, a.figures / "timing.png")
    fig_subjects(df, a.figures / "subjects.png")
    fig_preprocessing(df, a.figures / "preprocessing.png")
    t = tables(df)
    a.tables.write_text("\n\n".join(f"<!-- {k} -->\n{v}" for k, v in t.items()) + "\n")
    print("wrote", sorted(p.name for p in a.figures.glob("*.png")), "and", a.tables)


if __name__ == "__main__":
    main()
