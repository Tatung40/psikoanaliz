"""Python side of the independent R mirt comparison for unidimensional GRM/PCM/GPCM/NRM.

R itself runs in .github/workflows/r-mirt-polytomous.yml; here the formulas and the R -> Python mapping
are verified against Python mirt's own probability function (no R claimed).
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mirt")
from validation.r_mirt_polytomous.compare import MODELS, THETA, example, from_r, probabilities, python_fit


def _to_r(model, p):
    """Forward map (Python physical -> R slope-intercept), the inverse of from_r."""
    if model == "GRM":
        a = p["discrimination"]; d = -a[:, None] * p["thresholds"]
        return pd.DataFrame({"a1": a, **{f"d{k + 1}": d[:, k] for k in range(3)}})
    if model in ("PCM", "GPCM"):
        a = p["discrimination"]; d = np.column_stack([np.zeros(len(a)), -np.cumsum(a[:, None] * p["steps"], axis=1)])
        return pd.DataFrame({"a1": a, **{f"ak{k}": float(k) for k in range(4)}, **{f"d{k}": d[:, k] for k in range(4)}})
    a1 = p["slopes"][:, 3] / 3.0                      # R default ak3 = K-1 = 3, ak0 = 0
    return pd.DataFrame({"a1": a1, **{f"ak{k}": p["slopes"][:, k] / a1 for k in range(4)},
                         **{f"d{k}": p["intercepts"][:, k] for k in range(4)}})


@pytest.mark.parametrize("model", MODELS)
def test_formulas_reproduce_python_mirt_and_r_mapping_round_trips(model):
    _, data = example(model)
    fit, params, eap, psd = python_fit(model, data.to_numpy())
    library = np.asarray(fit.model.probability(THETA[:, None]), float).transpose(1, 0, 2)
    np.testing.assert_allclose(probabilities(model, params, THETA), library, rtol=0, atol=1e-12)
    back = from_r(model, _to_r(model, params))
    for key, value in params.items():
        np.testing.assert_allclose(back[key], value, rtol=0, atol=1e-12)
    assert fit.converged and np.isfinite(eap).all() and (psd > 0).all()
