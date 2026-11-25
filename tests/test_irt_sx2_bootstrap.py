"""Production bootstrap S-X2 p equals the preregistered validation algorithm; scope fails closed."""
import types
import numpy as np
from core import irt_sx2_bootstrap as P
from core.irt_python_mirt_adapter import fit_python_mirt
from validation.p3_sx2_bootstrap import run as V


def test_identical_to_validation_runner_and_fail_closed(monkeypatch):
    x, seed = V.data("3PL", 7, 0, False)
    out = fit_python_mirt(x, model="3PL", estimation="MAP", n_quadpts=61, max_iter=500, tol=1e-4)
    monkeypatch.setattr(V, "B", 4)
    ref = V.one(("3PL", 7, 0, False))
    p, reason = P.bootstrap_pvalues(out, x, seed=(7, 3, 0), n_boot=4)
    assert reason is None and np.allclose(p, ref["p_boot"], equal_nan=True)
    assert P.bootstrap_pvalues(types.SimpleNamespace(**{**out.__dict__, "n_items": 11}), x)[0] is None
