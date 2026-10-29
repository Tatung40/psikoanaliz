"""Python side of the independent R mirt 3PL/4PL comparison: the IRF/information formulas used to compare the
two engines reproduce Python mirt itself (R runs only in .github/workflows/r-mirt-34.yml)."""
import numpy as np
import pytest

pytest.importorskip("mirt")
from validation.r_mirt_34.compare import SEEDS, THETA, information, irf, params, python_fit, simulate


@pytest.mark.parametrize("model", sorted(SEEDS))
def test_formulas_reproduce_python_mirt_on_the_fixture(model):
    fit, table, reason = python_fit(model, simulate(model, SEEDS[model]))
    assert fit.converged and reason is None and (table.Not == "").all()     # fixture rule: interior solution
    p = params(table, model)
    np.testing.assert_allclose(irf(p, THETA), np.asarray(fit.model.probability(THETA[:, None])),
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(information(p, THETA), np.asarray(fit.model.information(THETA[:, None])),
                               rtol=1e-10, atol=1e-12)


def test_seed_rule_is_recorded_and_4pl_stays_unverified():
    """select_seeds() stopped at its pre-set limit: 3PL seed 13; no interior 4PL ML fit in seeds 1..50."""
    from validation.r_mirt_34.compare import compare
    assert SEEDS == {"3PL": 13}
    assert compare("4PL", None, {})["status"] == "NOT VERIFIED"
