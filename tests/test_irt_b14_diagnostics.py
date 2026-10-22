import numpy as np
import pandas as pd
from unittest.mock import patch
from core.irt_b14_diagnostics import model_diagnostics

class Fit:
    model=object()

def sample():
    return {"result":Fit(),"coded":pd.DataFrame(np.ones((20,4),dtype=int))}

def test_valid_item_diagnostics():
    with patch("mirt.itemfit",return_value=pd.DataFrame({"S_X2":[3.,4.]})),patch("mirt.compute_fit_indices",return_value={"M2":3.1}):
        status,tables=model_diagnostics(sample())
    assert len(tables)==2
    assert set(status["Durum"])=={"Motor çıktısı mevcut"}

def test_failed_diagnostics_no_invented_results():
    with patch("mirt.itemfit",side_effect=ValueError("unsupported")),patch("mirt.compute_fit_indices",side_effect=RuntimeError("failed")):
        status,tables=model_diagnostics(sample())
    assert not tables
    assert set(status["Durum"])=={"Hesaplanamadı"}
