"""NRM standard errors: the adapter must bypass the Python mirt 1.2.0 se_method='auto' NRM defect."""
import numpy as np
import pytest

mirt = pytest.importorskip("mirt")
from core.irt_python_mirt_adapter import fit_python_mirt
from validation.polytomous_mc_study import simulate


def test_adapter_nrm_uses_explicit_oakes_standard_errors():
    x, _ = simulate("NRM", 1500, 11)
    adapter = fit_python_mirt(x, model="NRM", n_categories=4).fit
    explicit = mirt.fit_mirt(x, model="NRM", n_categories=4, se_method="oakes")
    auto = mirt.fit_mirt(x, model="NRM", n_categories=4)
    for key in ("slopes", "intercepts"):
        a = np.asarray(adapter.standard_errors[key])[:, 1:]
        assert np.allclose(a, np.asarray(explicit.standard_errors[key])[:, 1:], rtol=1e-10)
        # the library default is roughly half as large (documented defect); guard against silent reversion
        assert np.median(np.asarray(auto.standard_errors[key])[:, 1:] / a) < .75


@pytest.mark.parametrize("model", ["GRM", "PCM", "GPCM"])
def test_other_families_unchanged_by_the_fix(model):
    x, _ = simulate(model, 800, 12)
    a = fit_python_mirt(x, model=model, n_categories=4).fit
    b = mirt.fit_mirt(x, model=model, n_categories=4)
    for key in a.standard_errors:
        assert np.allclose(np.asarray(a.standard_errors[key]), np.asarray(b.standard_errors[key]), equal_nan=True)
