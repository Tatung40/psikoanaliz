"""ICC(2,k) 95% CI = Spearman–Brown transform of the McGraw–Wong ICC(2,1) interval (coherent with ICC(A,k) = SB(ICC(A,1)))."""
import numpy as np


def test_icc2k_ci_is_spearman_brown_of_icc21_ci():
    from core.rater_reliability import _icc_ci
    rng = np.random.default_rng(5)
    for n, k in ((20, 4), (50, 3), (12, 6)):
        x = rng.normal(size=(n, 1)) + rng.normal(scale=.7, size=(n, k)) + rng.normal(scale=.3, size=(1, k))
        lo, hi = _icc_ci(x, "ICC(A,1)"); sb = lambda r: k * r / (1 + (k - 1) * r)
        np.testing.assert_allclose(_icc_ci(x, "ICC(A,k)"), [sb(lo), sb(hi)], rtol=0, atol=1e-12)


def test_evidence_enables_icc2k_ci_and_keeps_history():
    import json
    from pathlib import Path
    e = json.loads((Path(__file__).resolve().parents[1] / "validation/rater_reliability/evidence.json").read_text())["status"]["icc_2_k"]
    assert "ci95" in e["validated_outputs"] and e["historical_failed_outputs"] == ["ci95"] and "irr::icc" in e["ci95_resolution"]
