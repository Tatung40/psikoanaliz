"""Python side of the independent R mirt 3PL/4PL comparison: the IRF/information formulas used to compare the
two engines reproduce Python mirt itself (R runs only in .github/workflows/r-mirt-34.yml)."""
import numpy as np
import pytest

pytest.importorskip("mirt")
from validation.r_mirt_34.compare import MODELS, SEEDS, THETA, information, irf, params, python_fit, simulate


@pytest.mark.parametrize("model", MODELS)
def test_formulas_reproduce_python_mirt_on_the_fixture(model):
    fit, table, reason = python_fit(model, simulate(model, SEEDS[model]))
    assert fit.converged and reason is None and (table.Not == "").all()     # fixture rule: interior solution
    p = params(table, model)
    np.testing.assert_allclose(irf(p, THETA), np.asarray(fit.model.probability(THETA[:, None]))[:, :, 1],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(information(p, THETA), np.asarray(fit.model.information(THETA[:, None])),
                               rtol=1e-10, atol=1e-12)
