"""B13 high-resolution PNG images can be exported without browser renderer."""
import io
import pytest
from PIL import Image
from ui.irt_b13_view import native_fit
from reports.irt_b13_png import png_bytes
import numpy as np
import pandas as pd
from scipy.special import expit

@pytest.fixture(scope="module")
def result():
    rng=np.random.default_rng(25)
    th=rng.normal(size=120)
    b=np.linspace(-1,1,8)
    x=(rng.random((120,8))<expit(th[:,None]-b)).astype(int)
    return native_fit(pd.DataFrame(x,columns=[f"M{i}" for i in range(8)]),"2PL")

@pytest.mark.parametrize("kind",["ICC","IIC","TIF","SEM","Theta dağılımı","Wright haritası","Parametre dağılımı"])
def test_export_png(kind,result):
    raw=png_bytes(result,kind)
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    img=Image.open(io.BytesIO(raw))
    assert img.width>=1800 and img.height>=1000
