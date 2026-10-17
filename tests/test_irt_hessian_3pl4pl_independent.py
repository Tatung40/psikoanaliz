"""B2: independent check of the 3PL/4PL observed information and Hessian SEs.

Reference: direct four-point second differences of a marginal log-likelihood
written here with NumPy/SciPy only (no core likelihood or score code). The
module differentiates the analytic score instead, so agreement checks the
score, the parameter ordering and the information -> covariance -> SE chain.

Tolerances come from measured numerical stability (16 fixtures, both models,
N=400-2000, k=4-6, 21 nodes): the reference itself changes by <=5.1e-7
(relative to max|H|) between steps h and 2h; module-reference differences
were <=8.7e-8; score-Jacobian asymmetry <=1.2e-8; SE differences <=2.3e-6
while the reference SE changes by <=1.9e-6 between h and 2h. Tolerances are
about 4x-10x those measurements, far below the O(1) entry errors a wrong
derivative or ordering would cause.

Points are true generating values, not MLEs: in these small 3PL/4PL designs
fitted solutions reached a parameter bound (SE correctly refused), so an
interior MLE could not serve as fixture. SEs here verify numerics only and
carry no sampling interpretation.

B4: 4PL fixtures use k=5 (20 parameters <= 2**5-1 = 31). With k=4 the 4PL has
16 parameters for 15 pattern degrees of freedom and is structurally
unidentified; its observed information can still look positive definite, so
SEs are refused on structural grounds. No identifiable 4PL fixture with a
positive definite information matrix was found up to k=8, N=4000, so the
SE chain (model independent) is checked numerically on 3PL; 4PL-specific
derivatives are covered by the matrix tests here and by the mirt tests (B3).
"""
from functools import lru_cache
import numpy as np
import pandas as pd
import pytest
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit, logsumexp
from core.irt_simulation import simulate_irt
from core.irt_hessian import observed_information_se, _observed_information

POINTS=21
MATRIX_RTOL=2e-6
ASYMMETRY_RTOL=1e-7
SE_RTOL=2e-5
ACTIVE={"3PL":["a","b","c"],"4PL":["a","b","c","d"]}


def _items(model,design,k=None):
    k=k or (4 if model=="3PL" else 5)
    if design=="weak":
        a,b=np.linspace(1.,2.,k),np.linspace(-1,1,k)
        c,d=np.linspace(.12,.2,k),np.linspace(.85,.92,k)
    else:
        a,b=np.linspace(2.,2.6,k),np.linspace(-.4,.4,k)
        c,d=np.linspace(.15,.22,k),np.linspace(.82,.88,k)
    return pd.DataFrame({"Madde":[f"M{i+1}" for i in range(k)],"a":a,"b":b,"c":c,
                         "d":np.ones(k) if model=="3PL" else d})


def _loglik(x,a,b,c,d):
    g,w=hermgauss(POINTS)
    theta=np.sqrt(2.)*g
    p=c+(d-c)*expit(a*(theta[:,None]-b))
    return float(logsumexp(x@np.log(p).T+(1-x)@np.log1p(-p).T
                           +np.log(w/np.sqrt(np.pi)),axis=1).sum())


def _reference_information(x,items,model,scale=1e-4):
    act=ACTIVE[model]; m=len(act)
    v=items[act].to_numpy(float).reshape(-1)
    def f(z):
        t=items.copy(); t[act]=z.reshape(-1,m)
        return _loglik(x,*(t[q].to_numpy(float) for q in "abcd"))
    h=scale*np.maximum(1.,np.abs(v)); n=len(v); info=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            ei=np.zeros(n); ej=np.zeros(n); ei[i]=h[i]; ej[j]=h[j]
            info[i,j]=info[j,i]=-(f(v+ei+ej)-f(v+ei-ej)-f(v-ei+ej)+f(v-ei-ej))/(4*h[i]*h[j])
    return info


@lru_cache(maxsize=None)
def _case(model,design,n):
    items=_items(model,design)
    data,_=simulate_irt(n,items,seed=2026)
    x=data.to_numpy(float)
    raw,reason=_observed_information(x,items,ACTIVE[model],POINTS,4.,.65)
    assert not reason
    return items,data,raw,_reference_information(x,items,model)


CASES=[(m,design,n) for m in ("3PL","4PL") for design,n in (("weak",400),("strong",2000))]


@pytest.mark.parametrize("model,design,n",CASES)
def test_information_matches_direct_loglik_second_differences(model,design,n):
    _,_,raw,ref=_case(model,design,n)
    scale=np.abs(ref).max()
    assert np.abs(raw-raw.T).max()<=ASYMMETRY_RTOL*scale
    info=(raw+raw.T)/2
    assert np.abs(info-ref).max()<=MATRIX_RTOL*scale
    gap=np.linalg.norm(info-ref,2)
    assert np.abs(np.linalg.eigvalsh(info)-np.linalg.eigvalsh(ref)).max()<=gap*(1+1e-6)+1e-9


@pytest.mark.parametrize("model,design,n",[("3PL","weak",400),("4PL","weak",400),
                                            ("4PL","strong",2000)])
def test_indefinite_information_is_refused_consistently(model,design,n):
    items,data,_,ref=_case(model,design,n)
    assert np.linalg.eigvalsh(ref)[0]<0
    table,reason=observed_information_se(data,items,model,points=POINTS)
    assert table is None
    assert "pozitif tanımlı değil" in reason


@pytest.mark.parametrize("model",["3PL"])
def test_hessian_se_matches_independent_inverse(model):
    items,data,_,ref=_case(model,"strong",2000)
    eig=np.linalg.eigvalsh(ref)
    assert eig[0]>1e-7 and eig[-1]/eig[0]<1e8
    table,reason=observed_information_se(data,items,model,points=POINTS)
    assert not reason, reason
    act=ACTIVE[model]
    expected=np.sqrt(np.diag(np.linalg.inv(ref))).reshape(len(items),len(act))
    for h,q in enumerate(act):
        se=table[f"SH({q})"].to_numpy(float)
        np.testing.assert_allclose(se,expected[:,h],rtol=SE_RTOL,atol=0)
        est=items[q].to_numpy(float)
        np.testing.assert_allclose(table[f"{q} %95 alt"],est-1.96*se,rtol=0,atol=1e-12)
        np.testing.assert_allclose(table[f"{q} %95 üst"],est+1.96*se,rtol=0,atol=1e-12)


def test_item_order_permutes_standard_errors():
    items,data,_,_=_case("3PL","strong",2000)
    base,_=observed_information_se(data,items,"3PL",points=POINTS)
    order=[2,0,3,1]
    swapped,reason=observed_information_se(data.iloc[:,order],
                                           items.iloc[order].reset_index(drop=True),
                                           "3PL",points=POINTS)
    assert not reason
    assert list(swapped["Madde"])==list(items["Madde"].iloc[order])
    for q in ACTIVE["3PL"]:
        np.testing.assert_allclose(swapped[f"SH({q})"].to_numpy(),
                                   base[f"SH({q})"].to_numpy()[order],rtol=1e-8)


def test_structurally_unidentified_4pl_se_refused_despite_positive_definite_matrix():
    """4PL with k=4: 16 parameters > 15 pattern degrees of freedom."""
    items=_items("4PL","strong",k=4)
    data,_=simulate_irt(2000,items,seed=2026)
    raw,reason=_observed_information(data.to_numpy(float),items,ACTIVE["4PL"],POINTS,4.,.65)
    assert not reason
    eig=np.linalg.eigvalsh((raw+raw.T)/2)
    assert eig[0]>1e-7 and eig[-1]/eig[0]<1e8      # passes the numerical gates
    table,reason=observed_information_se(data,items,"4PL",points=POINTS)
    assert table is None
    assert "yapısal olarak tanımlanamaz" in reason
