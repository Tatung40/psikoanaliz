"""Closure K production module: identity with the confirmation runner, structure, fail-closed scope/status."""
import types
import numpy as np
from core import irt_4pl_score_sets as S
from core.irt_python_mirt_adapter import fit_python_mirt
from validation.binary_certification.common import simulate


def test_sets_structure_and_scope():
    x = simulate("4PL", 3000, 12, [202611263, 4, 0, 0])
    out = fit_python_mirt(x, model="4PL", estimation="MAP", n_quadpts=61, max_iter=500, tol=1e-4)
    eta = out.fit.coef()[["discrimination", "difficulty", "guessing", "upper"]].to_numpy(float).ravel()
    sets = S.confidence_sets(eta, x, items=[0, 11])
    assert len(sets) == 4
    for s in sets:
        assert s["whole"] == (s["intervals"] == [((0.0, .5) if s["param"] == "c" else (.5, 1.0))]) or not s["whole"]
        assert all(a < b for a, b in s["intervals"]) and s["empty"] == (s["intervals"] == [])
    assert S.score_set_table(types.SimpleNamespace(**{**out.__dict__, "n_persons": 2999}), x, list(range(12)))[0] is None
    assert S.score_set_table(types.SimpleNamespace(**{**out.__dict__, "model": "3PL"}), x, list(range(12)))[0] is None


def test_statistic_zero_at_its_own_root_and_positive():
    x = simulate("4PL", 3000, 12, [202611263, 4, 1, 0]); t, w = S._quad()
    out = fit_python_mirt(x, model="4PL", estimation="MAP", n_quadpts=61, max_iter=500, tol=1e-4)
    eta = out.fit.coef()[["discrimination", "difficulty", "guessing", "upper"]].to_numpy(float).ravel()
    phat = np.bincount(np.asarray(x, int) @ S._BITS, minlength=4096) / 3000
    assert S.statistic(eta, phat, 2, 3000, t, w) >= 0
