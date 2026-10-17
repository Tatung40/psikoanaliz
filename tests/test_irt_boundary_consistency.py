"""B1 regression: boundary flags agree with SE diagnostics."""
import inspect
from functools import lru_cache
import numpy as np
import pandas as pd
import pytest
from core.irt import fit_irt, sx2_pilot
from core.irt_simulation import simulate_irt
from core.irt_se_diagnostics import hessian_boundary_diagnostics, BOUNDARY_TOLERANCE
from core.irt_hessian import observed_information_se
from core.irt_stationarity import stationarity_diagnostics

COLUMNS=["Madde","a","b","c","d","Gözlenen p","Model p","Artık RMS","Sınır uyarısı"]

@lru_cache(maxsize=None)
def _data():
    truth=pd.DataFrame({"Madde":[f"M{i}" for i in range(6)],
                        "a":np.linspace(.8,1.6,6),"b":np.linspace(-1,1,6),
                        "c":0.,"d":1.})
    x,_=simulate_irt(300,truth,seed=3)
    return x

@lru_cache(maxsize=None)
def _fit(model):
    return fit_irt(_data(),model,points=21,max_iter=70)

@pytest.mark.parametrize("model,param,side",[("3PL","c","Alt sınır"),("4PL","d","Üst sınır")])
def test_boundary_flag_equals_hessian_diagnostics(model,param,side):
    fit=_fit(model)
    items=fit["items"]
    diag=hessian_boundary_diagnostics(items,model,a_upper=fit["a_upper"],d_lower=fit["d_lower"])
    assert ((diag["Parametre"]==param)&(diag["Sorun"]==side)).any()
    flagged=set(items.loc[items["Sınır uyarısı"],"Madde"])
    assert flagged==set(diag["Madde"])
    assert items["Sınır uyarısı"].dtype==bool

@pytest.mark.parametrize("model",["3PL","4PL"])
def test_sx2_pilot_gives_no_p_value_for_boundary_items(model):
    fit=_fit(model)
    items=fit["items"]
    table=sx2_pilot(_data(),items,model)
    boundary=items["Sınır uyarısı"].to_numpy()
    assert boundary.any()
    assert table.loc[boundary,"p (yaklaşık)"].isna().all()
    assert table.loc[boundary,"S-X² (pilot)"].isna().all()
    assert table.loc[boundary,"Durum"].str.contains("sınır").all()

@pytest.mark.parametrize("model",["3PL","4PL"])
def test_flagged_items_are_exactly_those_blocking_hessian_se(model):
    fit=_fit(model)
    _,reason=observed_information_se(_data(),fit["items"],model,points=21,
                                     a_upper=fit["a_upper"],d_lower=fit["d_lower"])
    assert fit["items"]["Sınır uyarısı"].any()
    assert "sınır" in reason

@pytest.mark.parametrize("model",["1PL","2PL","3PL","4PL"])
def test_output_columns_unchanged(model):
    assert list(_fit(model)["items"].columns)==COLUMNS

@pytest.mark.parametrize("model",["1PL","2PL"])
def test_one_and_two_pl_keep_legacy_rule(model):
    fit=_fit(model)
    t=fit["items"]
    legacy=(((model!="1PL")&((t["a"]<.17)|(t["a"]>fit["a_upper"]-.05)))|(t["b"].abs()>3.95))
    assert t["Sınır uyarısı"].tolist()==legacy.tolist()

def test_boundary_tolerance_has_single_source():
    assert inspect.signature(hessian_boundary_diagnostics).parameters["tolerance"].default==BOUNDARY_TOLERANCE
    assert inspect.signature(stationarity_diagnostics).parameters["boundary_tolerance"].default==BOUNDARY_TOLERANCE
