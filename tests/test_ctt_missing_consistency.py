"""CTT missing data: never scored as incorrect; total-based statistics on complete responders; independent formulas."""
import numpy as np
import pandas as pd
from scipy import stats
from core.item_analysis import compute_item_statistics
from core.reliability import compute_reliability_indices
from core.scoring import calculate_total_scores


def data():
    rng = np.random.default_rng(3); th = rng.normal(size=300); b = np.linspace(-1, 1, 8)
    x = pd.DataFrame((rng.random((300, 8)) < 1 / (1 + np.exp(-(th[:, None] - b)))).astype(float), columns=[f"m{j}" for j in range(8)])
    x.iloc[rng.choice(300, 40, replace=False), 2] = np.nan; x.iloc[rng.choice(300, 25, replace=False), 5] = np.nan
    return x


def test_item_statistics_use_observed_p_and_complete_case_totals():
    x = data(); cols = list(x)
    tot = calculate_total_scores(x, cols, missing_rule="strict")
    res, _ = compute_item_statistics(x, cols, tot, is_dichotomous=True)
    r = res.set_index("Madde")
    assert abs(r.loc["m2", "Güçlük (p)"] - x["m2"].mean()) < 1e-4            # observed responses, not missing-as-0
    assert r.loc["m2", "N"] == x["m2"].notna().sum()
    cc = x.dropna(); rest = cc.drop(columns="m0").sum(1)
    assert abs(r.loc["m0", "r_ir (Düzeltilmiş)"] - stats.pearsonr(cc["m0"], rest)[0]) < 1e-4
    zero = compute_item_statistics(x.fillna(0), cols, x.fillna(0).sum(1), is_dichotomous=True)[0].set_index("Madde")
    assert abs(zero.loc["m2", "Güçlük (p)"] - r.loc["m2", "Güçlük (p)"]) > 1e-3   # the two treatments genuinely differ
    internal, _ = compute_item_statistics(x, cols, is_dichotomous=True)             # default totals: no partial sums
    assert np.allclose(internal["r_ir (Düzeltilmiş)"], res["r_ir (Düzeltilmiş)"], equal_nan=True)


def test_reliability_listwise_matches_independent_formulas():
    x = data(); cc = x.dropna(); k = cc.shape[1]; tv = cc.sum(1).var(ddof=1)
    alpha = k / (k - 1) * (1 - cc.var(ddof=1).sum() / tv)
    n = len(cc); p = cc.mean(); kr20 = k / (k - 1) * (1 - n / (n - 1) * (p * (1 - p)).sum() / tv)   # sample-variance KR-20 (== alpha for 0/1)
    m = cc.sum(1).mean(); kr21 = k / (k - 1) * (1 - m * (k - m) / (k * tv))
    rel = compute_reliability_indices(x, list(x), is_dichotomous=True)
    assert abs(rel["cronbach_alpha"] - alpha) < 1e-6 and abs(rel["kr20"] - kr20) < 1e-6
    if rel.get("kr21") is not None: assert abs(rel["kr21"] - kr21) < 1e-6


def test_all_missing_rows_do_not_crash_or_zero_fill():
    x = data(); x.iloc[:, 0] = np.nan
    res, _ = compute_item_statistics(x, list(x), is_dichotomous=True)
    assert res["r_ir (Düzeltilmiş)"].isna().all()
