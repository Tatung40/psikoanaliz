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
