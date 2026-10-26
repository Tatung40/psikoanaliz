"""No-fit regression for the actual Plotly6 JSON -> browser precheck boundary."""
from pathlib import Path
import base64
import numpy as np
import pandas as pd
import plotly.io as pio
import pytest
from validation.r_mirt_phase2.check_wright_browser import vector

@pytest.mark.parametrize('stem',['rasch','lsat6'])
def test_actual_saved_plotly_vectors_match_exported_tables(stem):
    root=Path('validation/r_mirt_phase2/evidence')
    fig=pio.from_json((root/f'{stem}_wright.json').read_text())
    items=pd.read_csv(root/f'{stem}_items.csv');hist=pd.read_csv(root/f'{stem}_histogram.csv')
    assert isinstance(fig.data[1].y,dict)  # exact original failure, not a fake fit.
    np.testing.assert_allclose(vector(fig.data[1].y),items.b,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(vector(fig.data[0].y),hist.Merkez,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(-vector(fig.data[0].x),hist['Kişi sayısı'])
    assert list(fig.data[1].text)==items.Madde.tolist()
    assert vector([1.,2.]).tolist()==[1.,2.]
    with pytest.raises(ValueError):vector({'dtype':'O','bdata':base64.b64encode(b'12345678').decode()})
