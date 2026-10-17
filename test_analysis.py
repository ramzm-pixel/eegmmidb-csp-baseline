import numpy as np
import pandas as pd

from analysis import (PRIOR_WEIGHT, evaluate, lda, make_subject, predict_with_blended_means)

CALIB_METHODS = {"calib_scratch", "calib_newlda", "calib_adapt_center", "calib_adapt_means",
                 "calib_ref_ea", "calib_ref_ea_center"}


def synthetic_subjects(n_subjects=12, n_trials=42, n_ch=16, n_samples=200, seed=0):
    """Class effect = variance increase in one of two channels; subjects differ ~20x in amplitude and spatial mixing."""
    rng = np.random.default_rng(seed)
    subs = {}
    for i in range(n_subjects):
        y = np.tile([0, 1], n_trials // 2)
        run = np.repeat([1, 2, 3], n_trials // 3)
        X = rng.standard_normal((n_trials, n_ch, n_samples))
        X[y == 1, 0] *= 1.8
        X[y == 0, 1] *= 1.8
        mixing = np.eye(n_ch) + 0.3 * rng.standard_normal((n_ch, n_ch))
        X = np.einsum("pq,eqt->ept", mixing, X) * 10 ** rng.uniform(-1, 1)
        subs[f"S{i:03d}"] = make_subject(X * 1e-6, y, run)
    return subs


def test_alignment_fixes_cross_subject_scale():
    df = pd.DataFrame(evaluate(synthetic_subjects(), "w", "c", "t"))
    mean = df.groupby("method").acc.mean()
    assert mean["within_loro"] > 0.9
    assert mean["loso_ea"] > mean["loso_raw"] + 0.1


def test_all_arms_present_and_valid():
    df = pd.DataFrame(evaluate(synthetic_subjects(8), "w", "c", "t"))
    assert CALIB_METHODS <= set(df.method)
    assert df.acc.between(0, 1).all()
    assert set(df[df.method.isin(CALIB_METHODS)].k) == {2, 5, 10}


def test_calibration_only_mode_skips_zero_shot_rows():
    df = pd.DataFrame(evaluate(synthetic_subjects(8), "w", "c", "t", calibration_only=True))
    assert set(df.method) == CALIB_METHODS


def test_blended_means_equals_pooled_lda_when_prior_dominates():
    """With an enormous prior weight the target's own means barely count, so predictions match the pooled LDA."""
    rng = np.random.default_rng(1)
    y = np.tile([0, 1], 200)
    F = rng.standard_normal((400, 6)) + y[:, None] * 0.8
    pooled = lda().fit(F, y)
    F_test = rng.standard_normal((50, 6)) + 0.4
    fake_target_means = rng.standard_normal((2, 6)) * 5
    pred = predict_with_blended_means(pooled, F_test, fake_target_means, k=5, prior_weight=1e12)
    assert np.array_equal(pred, pooled.predict(F_test))


def test_blended_means_follow_target_when_prior_is_tiny():
    """With a tiny prior weight the target's own class means decide, so shifted data is classified correctly."""
    rng = np.random.default_rng(2)
    y = np.tile([0, 1], 200)
    F = rng.standard_normal((400, 3)) + np.where(y[:, None] == 1, 1.0, -1.0)
    pooled = lda().fit(F, y)
    means_target = np.array([[-1.0] * 3, [1.0] * 3])
    pred = predict_with_blended_means(pooled, np.array([[-1.0] * 3, [1.0] * 3]), means_target, k=5, prior_weight=1e-9)
    assert pred.tolist() == [0, 1]


def test_results_do_not_depend_on_subject_order():
    """Each subject has its own random generator, so processing order must not change calibration draws."""
    subs = synthetic_subjects(8)
    a = pd.DataFrame(evaluate(subs, "w", "c", "t", calibration_only=True))
    b = pd.DataFrame(evaluate(dict(reversed(list(subs.items()))), "w", "c", "t", calibration_only=True))
    key = ["subject", "method", "k"]
    merged = a.merge(b, on=key, suffixes=("_a", "_b"))
    assert len(merged) == len(a) == len(b)
    assert np.allclose(merged.acc_a, merged.acc_b, atol=1e-9)
