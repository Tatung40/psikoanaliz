"""B12.2 fast tests isolate orchestration from expensive calibration."""
import pandas as pd
import numpy as np
from core import irt_b12_compare as comp


def _fit(loglik, converged=True):
    return {"model":"4PL","loglik":loglik,"converged":converged,
            "iterations":3,"history":[loglik-2, loglik],
            "items":pd.DataFrame([{"Madde":f"M{i}","a":1.2,"b":0.,
                                    "c":.18,"d":.93} for i in range(1,7)])}


def test_three_methods_and_audit(monkeypatch):
    monkeypatch.setattr(comp,"fit_irt",lambda data,**kw: _fit(-500 if kw.get("refine") else -510))
    monkeypatch.setattr(comp,"fit_irt_multistart",lambda data,**kw: {"fit":_fit(-505)})
    table, fits=comp.compare_estimators(np.ones((40,6)))
    assert table.method.tolist()==["EM","Multistart EM","EM + L-BFGS-B"]
    assert table.loglik.tolist()==[-510,-505,-500]
    assert table.status.eq("completed").all()
    assert len(fits)==3


def test_failure_is_reported_not_suppressed(monkeypatch):
    def fake(data, **kwargs):
        if kwargs.get("refine"):
            raise RuntimeError("optimizer failed")
        return _fit(-510, False)
    monkeypatch.setattr(comp,"fit_irt",fake)
    monkeypatch.setattr(comp,"fit_irt_multistart",lambda data,**kw: {"fit":_fit(-505)})
    table, fits=comp.compare_estimators(np.ones((40,6)))
    assert table.iloc[2]["status"]=="failed"
    assert "optimizer failed" in table.iloc[2]["error"]
    assert not bool(table.iloc[0]["converged"])
    assert len(fits)==2
