"""B14 categories are scored only if explicitly ordered; A-E stay nominal."""
import numpy as np
import pandas as pd
import pytest
from core.irt_b14 import prepare_polytomous

def nominal(n=90):
    letters=np.array(list("ABCDE"))
    x=np.resize(letters,n*5).reshape(n,5)
    return pd.DataFrame(x,columns=list("JKLMN"))

def test_nominal_ae_preserved_as_distinct_categories():
    x,lab=prepare_polytomous(nominal(),"NRM")
    assert lab==("A","B","C","D","E")
    assert set(x.stack().unique())==set(range(5))
    assert x.iloc[0,0]==0 and x.iloc[0,2]==2

def test_reject_nominal_letters_in_ordered_model():
    with pytest.raises(ValueError):
        prepare_polytomous(nominal(),"GRM")

@pytest.mark.parametrize("model",["GRM","PCM","GPCM","RSM"])
def test_ordered_scores(model):
    x=pd.DataFrame(np.resize(np.arange(4),360).reshape(90,4))
    coded,labs=prepare_polytomous(x,model)
    assert labs==("0","1","2","3")
    assert coded.shape==(90,4)

def test_reject_nominal_unknown_symbol():
    x=nominal();x.iloc[0,0]="F"
    with pytest.raises(ValueError,match="tanımlı olmayan"):
        prepare_polytomous(x,"NRM")

def test_real_grm_python_mirt():
    mirt=pytest.importorskip("mirt")
    from core.irt_b14 import fit_b14
    x=mirt.simdata(model="GRM",n_persons=100,n_items=5,n_categories=4,seed=94)
    res=fit_b14(pd.DataFrame(x),"GRM")
    assert res["engine"]=="Python mirt 1.2.0"
    assert res["result"] is not None

def test_streamlit_navigation():
    st=pytest.importorskip("streamlit.testing.v1")
    from pathlib import Path
    at=st.AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py"),default_timeout=120)
    d=nominal()
    for k,v in {"raw_df":d,"scored_df":None,"item_cols":list(d.columns),
                "id_col":None,"group_col":None,"nav_page":"MTK: Çok Kategorili B14"}.items():
        at.session_state[k]=v
    at.run()
    assert not at.exception
    assert any(b.key=="b14_run" for b in at.button)
