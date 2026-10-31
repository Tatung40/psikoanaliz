"""No fits: correlated intervals must not create fictitious MC precision."""
import numpy as np
import pytest

from validation.coverage_precision import coverage_precision


def test_perfectly_correlated_items_have_replication_not_interval_precision():
    # Independent datasets: two entirely covered, two entirely missed.
    parts = [np.full((30, 1), value) for value in (1, 1, 0, 0)]
    result = coverage_precision(parts, 4)
    # SE of four independent Bernoulli observations, with sample variance.
    assert result["coverage_mc_se"] == pytest.approx(np.sqrt(1 / 12))
    # Duplicating dependent items must not improve Monte Carlo precision.
    duplicated = coverage_precision([np.repeat(p, 10, axis=0) for p in parts], 4)
    assert duplicated["coverage_mc_se"] == pytest.approx(result["coverage_mc_se"])


def test_variable_eligibility_uses_ratio_and_retains_failed_replications():
    parts = [np.array([[0, 1], [0, 1]]), np.array([[0, 0], [1, 1]])]
    result = coverage_precision(parts, 3, eligible_column=0)
    # Totals covered=(2,0,0), eligible=(2,1,0); pooled coverage=2/3.
    assert result["coverage_mc_se"] == pytest.approx(np.sqrt(4 / 27))
    assert result["independent_replications"] == 3
    assert result["eligible_replications"] == 2
    assert result["coverage_cluster_counts"] == {"covered": [2, 0, 0], "eligible": [2, 1, 0]}


@pytest.mark.parametrize("parts,reps,column", [([], 3, None), ([np.ones((3, 1))], 1, None),
                                              ([np.ones((3, 2))], 3, 0)])
def test_unestimable_precision_is_explicit(parts, reps, column):
    assert coverage_precision(parts, reps, eligible_column=column)["coverage_mc_se"] is None


@pytest.mark.parametrize("parts,reps,column", [([np.ones((2, 1))], 0, None),
    ([np.array([[.5]])], 2, None), ([np.array([[2, 1]])], 2, 0)])
def test_invalid_counts_and_indicators_are_rejected(parts, reps, column):
    with pytest.raises(ValueError):
        coverage_precision(parts, reps, eligible_column=column)


@pytest.mark.parametrize("study", ["binary", "binary34"])
def test_study_paths_report_cluster_precision_without_calibration(monkeypatch, study):
    from types import SimpleNamespace
    import pandas as pd
    from core import irt_binary_inference, irt_python_mirt_adapter
    from validation import binary_se_coverage_study as binary
    from validation import binary34_se_coverage_study as binary34

    monkeypatch.setattr(irt_python_mirt_adapter, "fit_python_mirt",
                        lambda *a, **kw: SimpleNamespace(fit=SimpleNamespace(converged=True)))
    outcomes = iter((True, True, False, False))
    truth = np.linspace(-1.5, 1.5, 2) if study == "binary" else np.linspace(-1.25, 1.25, 2)

    def inference(*args):
        shift = 0 if next(outcomes) else 2
        return pd.DataFrame({"Madde": ["M0", "M1"], "Parametre": ["b", "b"],
                             "Kestirim": truth, "GA %95 alt": truth - 1 + shift,
                             "GA %95 üst": truth + 1 + shift}), None

    monkeypatch.setattr(irt_binary_inference, "parameter_inference", inference)
    if study == "binary":
        monkeypatch.setattr(binary, "simulate", lambda *a: np.array([[0, 0], [1, 1]]))
        result = binary.run({**binary.DESIGN, "models": ["1PL"], "items": 2, "replications": 4})["1PL"]
    else:
        result = binary34._cell(("3PL", "MAP", 4, 2, 4, 0))
    assert result["b"]["coverage"] == .5
    assert result["b"]["coverage_mc_se"] == pytest.approx(np.sqrt(1 / 12))
    assert result["converged"] == result["replications"] == 4


def test_constant_item_skips_are_counted_without_changing_coverage(monkeypatch):
    from types import SimpleNamespace
    import pandas as pd
    from core import irt_binary_inference, irt_python_mirt_adapter
    from validation import binary_se_coverage_study as binary
    samples = iter((np.zeros((2, 2)), np.array([[0, 0], [1, 1]]), np.ones((2, 2))))
    monkeypatch.setattr(binary, "simulate", lambda *a: next(samples))
    monkeypatch.setattr(irt_python_mirt_adapter, "fit_python_mirt",
                        lambda *a, **kw: SimpleNamespace(fit=SimpleNamespace(converged=True)))
    truth = np.linspace(-1.5, 1.5, 2)
    table = pd.DataFrame({"Parametre": ["b", "b"], "Kestirim": truth,
                          "GA %95 alt": truth-1, "GA %95 üst": truth+1})
    monkeypatch.setattr(irt_binary_inference, "parameter_inference", lambda *a: (table, None))
    result = binary.run({**binary.DESIGN, "models": ["1PL"], "items": 2, "replications": 3})["1PL"]
    assert result["constant_item_skipped"] == 2
    assert result["replications"] == 3 and result["converged"] == 1 and result["se_withheld"] == 0
    assert result["b"]["coverage"] == 1 and result["b"]["intervals"] == 2
    assert result["b"]["coverage_cluster_counts"] == {"covered": [2, 0, 0], "eligible": [2, 0, 0]}
    assert not binary.judge({"1PL": result})["1PL_convergence"]
