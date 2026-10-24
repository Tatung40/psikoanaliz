"""Known-truth generator contracts independent of the mirt fit engine."""
import numpy as np
import pytest
from validation.known_truth_irt_recovery import simulate, truth, MODELS


@pytest.mark.parametrize("model", MODELS)
def test_known_truth_generator_reproducible(model):
    x1, p1 = simulate(model, n_persons=80, n_items=7, seed=2026)
    x2, p2 = simulate(model, n_persons=80, n_items=7, seed=2026)
    np.testing.assert_array_equal(x1, x2)
    assert x1.shape == (80, 7)
    assert set(np.unique(x1)).issubset({0, 1})
    for key in ("a", "b", "c", "d"):
        np.testing.assert_array_equal(p1[key], p2[key])


@pytest.mark.parametrize("model", MODELS)
def test_known_truth_constraints(model):
    params = truth(model, 12)
    assert np.all(params["a"] > 0)
    assert np.all(params["c"] >= 0)
    assert np.all(params["d"] <= 1)
    assert np.all(params["c"] < params["d"])
    if model == "1PL":
        assert len(np.unique(params["a"])) == 1
    if model in ("1PL", "2PL"):
        assert np.all(params["c"] == 0)
    if model != "4PL":
        assert np.all(params["d"] == 1)


def test_known_truth_invalid_model_fails_closed():
    with pytest.raises(ValueError):
        simulate("5PL")


def test_recovery_summary_counts_failures_and_aggregates_only_measured():
    from validation.known_truth_irt_recovery import summarize
    records = [
        {"model": "1PL", "status": "MEASURED", "metrics": {
            "a": {"bias": 0.2, "mae": 0.2, "rmse": 0.2, "max_absolute_error": 0.2}}},
        {"model": "1PL", "status": "FAILED", "metrics": {}},
        {"model": "2PL", "status": "NONCONVERGED", "metrics": {}},
    ]
    summary = summarize(records)
    assert summary["1PL"]["attempted"] == 2
    assert summary["1PL"]["measured"] == 1
    assert summary["1PL"]["failed"] == 1
    assert summary["1PL"]["parameters"]["a"]["mae"] == pytest.approx(0.2)
    assert summary["2PL"]["nonconverged"] == 1
    assert summary["3PL"]["measured"] == 0


def test_fitting_failure_is_reported_not_silently_discarded(monkeypatch):
    import validation.known_truth_irt_recovery as recovery

    def broken_fit(*args, **kwargs):
        raise RuntimeError("intentional calibration failure")

    monkeypatch.setattr(recovery.mirt, "fit_mirt", broken_fit)
    result = recovery.evaluate("2PL", n_persons=50, n_items=6)
    assert result["status"] == "FAILED"
    assert result["error_type"] == "RuntimeError"
    assert result["metrics"] == {}
    assert result["converged"] is None


def test_sensitivity_summary_reports_each_sample_size_without_real_fits(monkeypatch):
    import validation.known_truth_irt_recovery as recovery

    def fake_evaluate(model, *, n_persons, n_items, seed):
        return {"model": model, "n_persons": n_persons, "n_items": n_items,
                "seed": seed, "status": "MEASURED", "converged": True,
                "metrics": {p: {"mae": float(n_persons) / 1000}
                            for p in ("a", "b", "c") + (("d",) if model == "4PL" else ())}}

    monkeypatch.setattr(recovery, "evaluate", fake_evaluate)
    report = recovery.sensitivity_run(sample_sizes=(1000, 3000), seeds=(11, 12))
    assert len(report["runs"]) == 8
    assert report["summary_by_sample_size"]["3PL_N1000"]["measured"] == 2
    assert report["summary_by_sample_size"]["4PL_N3000"]["mae"]["d"] == pytest.approx(3.0)


def test_sensitivity_summary_never_hides_failed_runs(monkeypatch):
    import validation.known_truth_irt_recovery as recovery

    def failed(model, *, n_persons, n_items, seed):
        return {"model": model, "n_persons": n_persons, "n_items": n_items,
                "seed": seed, "status": "FAILED", "metrics": {}}

    monkeypatch.setattr(recovery, "evaluate", failed)
    report = recovery.sensitivity_run(sample_sizes=(1000,), seeds=(11,))
    for row in report["summary_by_sample_size"].values():
        assert row["attempted"] == 1 and row["failed"] == 1
        assert row["measured"] == 0 and row["mae"] == {}
