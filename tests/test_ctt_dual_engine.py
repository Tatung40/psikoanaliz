import io
import numpy as np
import pandas as pd
from docx import Document
from core.ctt_dual_engine import compare_ctt,item_guidance
from reports.ctt_guided_docx import ctt_docx

def test_two_alpha_engines_and_docx():
    rng=np.random.default_rng(8)
    latent=rng.normal(size=(150,1))
    x=pd.DataFrame((latent+rng.normal(size=(150,7))>0).astype(int),
                   columns=[f"Q{i}" for i in range(7)])
    r=compare_ctt(x,list(x.columns))
    assert r["reference_alpha"] is not None
    assert r["difference"]<2e-5
    assert len(item_guidance(r))==7
    doc=Document(io.BytesIO(ctt_docx(r)))
    assert "Cronbach" in " ".join(p.text for p in doc.paragraphs)

def test_missing_data_listwise():
    x=pd.DataFrame(np.random.default_rng(4).integers(0,2,(60,5)))
    x.loc[0,0]=np.nan
    r=compare_ctt(x,list(x.columns))
    assert r["N"]==59 and r["removed"]==1
