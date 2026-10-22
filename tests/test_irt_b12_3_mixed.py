"""Targeted B12.3 tests: reference likelihood, derivatives, validation safeguards."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from core import irt_mixed as X
from core.irt_mg import Quadrature, _parts, as_matrix

F=Path(__file__).parent/"fixtures"/"irt_b11"
TYPES=["1PL"]*3+["2PL"]*3+["3PL"]*3+["4PL"]*3
PARS=pd.DataFrame({"a":[1,1,1,.8,1.3,1.9,1.1,1.6,.9,1.4,2.,1.2],
                   "b":[-1,0,1,-.5,.4,1.2,-.2,.8,1.5,-.8,.1,1.],
                   "c":[0,0,0,0,0,0,.15,.22,.3,.1,.2,.05],
                   "d":[1,1,1,1,1,1,1,1,1,.9,.8,.95]})

def data():return pd.read_csv(F/"data_irt4pl_ok.csv")

def test_fixed_parameter_loglik_matches_mirt():
    assert X.loglik_at(data(),TYPES,PARS)==pytest.approx(-26454.2019664538,rel=1e-12)

@pytest.mark.parametrize("ng",[1,2])
def test_analytic_gradient_matches_finite_differences(ng):
    x,obs=as_matrix(data().iloc[:200])
    gv=(np.arange(len(x))%3==0).astype(int) if ng==2 else np.zeros(len(x),int)
    des=X.MixedDesign(TYPES,ng,group_specific=(4,10),prior={"c":("beta",5.,17.),"d":("beta",17.,5.)})
    parts=_parts(x,obs,gv,ng);q=Quadrature("mirt",61)
    v=X.start_values(x,obs,des,np.random.default_rng(1),.5)
    _,g=X.negloglik(v,des,parts,q,len(x))
    fd=[]
    for i in range(des.size):
        z=np.zeros_like(v);z[i]=1e-6
        fd.append((X.negloglik(v+z,des,parts,q,len(x))[0]-X.negloglik(v-z,des,parts,q,len(x))[0])/2e-6)
    assert np.max(np.abs(g-np.array(fd)))<1e-5

def test_group_and_quadrature_guards():
    d=data().iloc[:80]
    with pytest.raises(ValueError,match="referans"):
        X.fit_mixed(d,TYPES,groups=["A"]*80,reference="A",focal="A")
    with pytest.raises(ValueError,match="uzunluğu"):
        X.fit_mixed(d,TYPES,groups=["A"]*79,reference="A",focal="B")
    with pytest.raises(ValueError,match="Tanımsız"):
        X.fit_mixed(d,TYPES,groups=["A"]*40+["B"]*39+["C"],reference="A",focal="B")
    with pytest.raises(ValueError,match="61"):
        X.fit_mixed(d,TYPES,points=31)
    with pytest.raises(ValueError,match="Tek grupta DIF"):
        X.fit_mixed(d,TYPES,group_specific=["X"])
    with pytest.raises(ValueError,match="Bilinmeyen"):
        X.fit_mixed(d,TYPES,groups=["A"]*40+["B"]*40,reference="A",focal="B",group_specific=["X"])

def test_design_fixed_parameters():
    des=X.MixedDesign(TYPES)
    v=X.start_values(np.ones((35,12)),np.ones((35,12),bool),des)
    a,b,c,d=des.unpack(v,0)
    assert np.all(a[:3]==1.) and np.all(c[:6]==0.) and np.all(d[:9]==1.)

def test_incorrect_priors_rejected():
    with pytest.raises(ValueError,match="Önsel"):
        X.MixedDesign(TYPES,prior={"a":("beta",2.,2.)})
