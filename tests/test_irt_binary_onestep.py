"""Production one-step CI equals the independently confirmed validation kernel; scope fails closed."""
import types
import numpy as np
import pytest
from core import irt_binary_onestep as O
from validation.binary_certification.common import truth, simulate
from validation.p3_sx2_analytic import kernel as K


@pytest.mark.parametrize("model", ["3PL", "4PL"])
def test_information_and_prior_gradient_identical_to_confirmed_kernel(model):
    p = truth(model, 12); eta = np.stack([p[q] for q in "abcd"[:4 if model == "4PL" else 3]], 1).ravel()
    px, s = K.pattern_terms(eta, model); I = (px[:, None] * s).T @ s
    assert np.allclose(O.expected_information(eta, model), I, rtol=1e-10, atol=1e-14)
    assert np.allclose(O.log_prior_gradient(eta, model), K.log_prior_grad_hess(eta, model)[0], rtol=1e-12)


def test_real_fit_table_scope_and_fail_closed():
    from core.irt_python_mirt_adapter import fit_python_mirt
    x = simulate("3PL", 3000, 12, [202611259, 3, 0, 0])
    out = fit_python_mirt(x, model="3PL", estimation="MAP", n_quadpts=61, max_iter=500, tol=1e-4)
    table, reason = O.onestep_intervals(out, [f"I{j}" for j in range(12)])
    assert reason is None and len(table) == 36
    assert (table["%95 GA alt"] <= table["%95 GA üst"]).all() and (table.query("Parametre=='guessing'")[["%95 GA alt", "%95 GA üst"]].stack().between(0, .5)).all()
    assert np.allclose(table["MAP kestirimi"].to_numpy(), out.fit.coef()[["discrimination", "difficulty", "guessing"]].to_numpy().ravel())
    small = types.SimpleNamespace(**{**out.__dict__, "n_persons": 2999})
    assert O.onestep_intervals(small, list(range(12)))[0] is None
    ml = types.SimpleNamespace(**{**out.__dict__, "estimation": "ML"})
    assert O.onestep_intervals(ml, list(range(12)))[0] is None
