"""Nested LR maxima must not fabricate evidence of no DIF after a failed fit."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2
from core import irt_dif as binary


def fit(ll, **updates):
    return dict(converged=True, regular=True, linked=True, bound_labels=[],
                loglik=ll, cov=np.eye(1), cov_conditional=False, quadrature_ok=True, **updates)


@pytest.mark.parametrize("alt,null,expected", [
    (-100., -105., 10.), (-105., -105., 0.),
    (-105. - 1e-8, -105., 0.),
])
def test_valid_nested_lr_and_roundoff(alt, null, expected):
    assert binary._lr(fit(alt), fit(null)) == pytest.approx(expected)


@pytest.mark.parametrize("alt,null", [
    (-106., -105.), (-1e9 - .01, -1e9),  # absolute, not N-scaled tolerance
    (np.nan, -105.), (-100., np.nan), (np.inf, -105.),
    (-100., -np.inf), (-np.inf, -105.),
])
def test_invalid_nested_lr_has_no_p_value(alt, null):
    stat = binary._lr(fit(alt), fit(null))
    assert np.isnan(stat) and np.isnan(chi2.sf(stat, 1))


def test_nonconverged_fits_still_block_inference():
    alt = fit(-100.)
    alt["converged"] = False
    assert np.isnan(binary._lr(alt, fit(-105.)))


@pytest.mark.parametrize("model", ["1PL", "2PL"])
def test_failed_nested_maximum_is_disclosed_in_item_table(monkeypatch, model):
    class FailedMaximum:
        def __init__(self, *args):
            self.cache = {}

        def __call__(self, types, information=False):
            design = binary.Design(model, types, 2)
            params = np.zeros(design.size)
            for j, (kind, _, _) in enumerate(design.labels):
                if kind == "a":
                    params[j] = 1.
            # A purportedly converged free model with a worse objective.
            return dict(design=design, params=params, cov=None, se_reason="",
                        converged=True, loglik=-100. - sum(t != "anchor" for t in types),
                        mu_F=0., sd_F=1.)

    monkeypatch.setattr(binary, "_Fitter", FailedMaximum)
    data = pd.DataFrame(np.tile([[0, 1, 0], [1, 0, 1]], (100, 1)))
    result = binary.dif_analysis(data, ["R"] * 100 + ["F"] * 100,
                                 "R", "F", model=model, purify=False)
    table = result["table"]
    assert table["p"].isna().all() and table["q (BH)"].isna().all()
    assert not table["DIF işareti"].any()
    assert table["Not"].str.contains("optimizasyon doğrulanamadı").all()
