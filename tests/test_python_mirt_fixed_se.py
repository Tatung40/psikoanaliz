"""Fixed/identification-constrained Python mirt parameters must not report SE = 0."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mirt")
from core import example_datasets as ex
from core.irt_b14 import prepare_polytomous
from core.irt_python_mirt_adapter import fit_python_mirt
from core.irt_python_mirt_parameters import coef_with_se_table


def _fit(model):
    spec = next(d for d in ex.registry().values() if d.family == "irt" and d.model == model)
    bundle = ex.materialize(spec.id)
    labels = ["A", "B", "C", "D"] if model == "NRM" else None
    x, _ = prepare_polytomous(bundle["data"][bundle["items"]], model, categories=labels)
    return fit_python_mirt(x.to_numpy(), model=model, n_categories=4).fit


@pytest.mark.parametrize("model,fixed", [("PCM", ["discrimination_se"]),
                                         ("NRM", ["slopes_1_se", "intercepts_1_se"])])
def test_fixed_parameter_se_is_missing_not_zero(model, fixed):
    fit = _fit(model)
    raw = fit.coef_with_se()
    table = coef_with_se_table(fit)
    for column in fixed:
        assert (raw[column] == 0).all()
        assert table[column].isna().all()
    estimated = [c for c in table if c.endswith("_se") and c not in fixed]
    assert estimated and np.isfinite(table[estimated].to_numpy()).all()
    assert (table[estimated].to_numpy() > 0).all()
    pd.testing.assert_frame_equal(table[estimated], raw[estimated])


def test_b14_report_uses_masked_se_table():
    from reports.irt_b14_report import report_tables
    fit = _fit("PCM")
    spec = next(d for d in ex.registry().values() if d.family == "irt" and d.model == "PCM")
    bundle = ex.materialize(spec.id)
    coded, labels = prepare_polytomous(bundle["data"][bundle["items"]], "PCM")
    result = {"result": fit, "coded": coded, "labels": labels, "model": "PCM", "items": bundle["items"],
              "engine": "python_mirt", "status": "", "N": len(coded), "k": coded.shape[1]}
    frames = report_tables(result)
    assert frames["Motor Parametreleri"]["discrimination_se"].isna().all()
