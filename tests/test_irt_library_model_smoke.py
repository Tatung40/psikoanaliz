"""Independent deterministic external-library calibration smoke tests.

These tests prove invocation and input handling, NOT parameter recovery or
model certification. The simulated sample is intentionally small for CI.
"""
import numpy as np
import pytest
from core.irt_python_mirt_adapter import fit_python_mirt


@pytest.mark.parametrize("model", ["1PL", "2PL", "3PL", "4PL"])
def test_real_library_fit_all_binary_models(model):
    mirt = pytest.importorskip("mirt")
    responses = mirt.simdata(model=model, n_persons=100, n_items=8, seed=731)
    result = fit_python_mirt(responses, model=model)
    assert result.engine == "python_mirt"
    assert result.model == model
    assert result.n_persons == 100
    assert result.n_items == 8
    assert result.fit is not None
    assert result.scientific_status.startswith("EXPERIMENTAL")


@pytest.mark.parametrize("model", ["PCM", "GPCM", "NRM"])
def test_real_library_fit_polytomous_models(model):
    mirt = pytest.importorskip("mirt")
    responses = mirt.simdata(model=model, n_persons=100, n_items=8,
                             n_categories=4, seed=731)
    result = fit_python_mirt(responses, model=model, n_categories=4)
    assert result.fit is not None
    assert result.model == model
