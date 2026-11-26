import math
import numpy as np
from scipy import integrate,stats
from scipy.special import expit
from validation.p3_4pl_closure.dgp import data,distribution,first_probability,truth_eta
from validation.p3_4pl_closure.diagnostics import independent_old_bound
from validation.p3_empirical import kernel as E

def test_old_bound_independently_integrates_and_uses_normal_mgf():
    d=independent_old_bound()
    mgf=integrate.quad(lambda z:np.exp(-5*z)*stats.norm.pdf(z),-20,15,epsabs=1e-6)[0]
    assert abs(mgf/math.exp(12.5)-1)<1e-10
    assert abs(d["power_upper"]-(.05+3000*.8*np.exp(-30)*mgf))<1e-13
    assert d["adaptive_actual_disagreement"]<d["analytic_per_person"]
    assert abs(d["power_upper"]-.05006026397973786)<1e-14

def test_exact_pattern_law_and_native_null_are_consistent():
    p,_=E.terms(truth_eta(),"4PL",121)
    assert np.max(abs(p-distribution("null",121)))<1e-14
    for kind in ("nonmonotone","local_dependence","two_stage"):
        q=distribution(kind,121)
        assert abs(q.sum()-1)<1e-12 and (q>0).all()
        assert np.linalg.norm(q-p)>1e-4
        x,seed=data(202611264091,0,kind,N=80000)
        assert abs(x[:,0].mean()-q@E.X[:,0])<.007
        assert seed==[202611264091,4,0,("null","nonmonotone","local_dependence","two_stage").index(kind)]

def test_departures_are_real_not_relabelled_logistic():
    theta=np.linspace(-4,4,2001);p=first_probability(theta,"nonmonotone")
    assert np.diff(p).min()<-.0001 and np.diff(p).max()>.0001
    a=first_probability(theta,"local_dependence",1);b=first_probability(theta,"local_dependence",0)
    assert (a>b).all() and (a-b).max()>.1
    q=first_probability(theta,"two_stage")
    L=np.log((q-.1)/(.9-q))
    assert np.diff(q).min()>0 and abs(np.diff(L,2)).max()>1e-5

def test_worst_case_failures_never_disappear():
    from validation.p3_4pl_closure.summary import summarize
    rows=[dict(kind="null",r=i,error="failure") for i in range(3)]
    rows+=[dict(kind=k,r=0,error="failure") for k in ("nonmonotone","local_dependence","two_stage")]
    r=summarize(rows,"p_constrained","pilot")
    assert r["TypeI"]==1 and r["refusal"]==1 and r["certified_claims"]==[]
    assert all(v["power"]==0 and v["refusal"]==1 for v in r["power"].values())

def test_monotone_kl_projection_matches_independent_constrained_optimizer():
    from scipy import optimize
    from scipy.special import xlogy
    from validation.p3_4pl_closure.witness import pav,bound
    v=np.array([.1,.4,.3,.8,.6]);w=np.array([.1,.2,.3,.1,.3])
    q=pav(v,w)
    obj=lambda p:float(-w@(xlogy(v,p)+xlogy(1-v,1-p)))
    r=optimize.minimize(obj,np.linspace(.2,.7,5),method="SLSQP",bounds=[(.001,.999)]*5,
        constraints=[{"type":"ineq","fun":lambda p:np.diff(p)}],options={"ftol":1e-13,"maxiter":500})
    assert r.success and np.max(abs(r.x-q))<1e-6
    for points in (61,121):
        assert bound(distribution("null",points))["KL_global_lower"]<1e-12
    # A nonzero bound is evidence, never assumed merely because a curve is nonlogistic.
    b=bound(distribution("nonmonotone",121))
    assert b["KL_global_lower"]>=0
