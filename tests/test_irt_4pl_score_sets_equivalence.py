"""Optimised closure-K kernel equals the frozen certified implementation (validation/p1_4pl_score_perf)."""
import importlib.util
import numpy as np
from core import irt_4pl_score_sets as NEW
from validation.binary_certification.common import simulate, truth

spec = importlib.util.spec_from_file_location("frozen_k", "validation/p1_4pl_score_perf/frozen_k.py")
OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)


def test_terms_identity_including_boundaries():
    t, w = NEW._quad(); p = truth("4PL", 12); base = np.stack([p[s] for s in "abcd"], 1).ravel()
    rng = np.random.default_rng(0)
    for k in range(6):
        e = base * (1 + .2 * rng.standard_normal(48)); e[0::4] = np.abs(e[0::4]) + .05
        e[2::4] = [1e-6, .4999, .2][k % 3]; e[3::4] = [.5 + 1e-6, 1 - 1e-6, .9][k % 3]
        a, b = OLD.terms(e, t, w), NEW.terms_fast(e, t, w)
        assert np.allclose(a[0], b[0], rtol=1e-12, atol=0) and np.allclose(a[1], b[1], rtol=1e-10, atol=1e-12)


def test_sets_identical_to_frozen_on_confirmation_case():
    x = simulate("4PL", 3000, 12, [202611262101, 4, 0, 0])
    p = truth("4PL", 12); eta = np.stack([p[s] for s in "abcd"], 1).ravel()
    assert NEW.confidence_sets(eta, x, items=[0, 11], truth=eta) == OLD.confidence_sets(eta, x, items=[0, 11], truth=eta)
