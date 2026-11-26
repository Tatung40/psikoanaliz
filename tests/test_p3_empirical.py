"""Independent deterministic checks; no calibration or certification simulations."""
import numpy as np
import pytest
from scipy import optimize,stats
from validation.binary_certification.common import truth,PRIORS
from validation.p3_empirical import kernel as E


def eta(model):
    p=truth(model,12);return np.stack([p[q] for q in "abcd"[:4 if model=="4PL" else 3]],1).ravel()


@pytest.mark.parametrize("model",["3PL","4PL"])
def test_analytic_joint_score_and_observed_hessian_two_steps(model):
    e=eta(model);px,s=E.terms(e,model)
    weights=px*(1+.05*np.sin(np.arange(len(px))));weights/=weights.sum()
    _,_,H=E.terms(e,model,61,weights)
    v=np.random.default_rng(41).normal(size=len(e));v/=np.linalg.norm(v)
    for step in (1e-5,5e-6):
        p1,s1=E.terms(e+step*v,model);p0,s0=E.terms(e-step*v,model)
        fd=(np.log(p1)-np.log(p0))/(2*step)
        assert np.allclose(fd,s@v,rtol=2e-5,atol=2e-8)
        Hv=weights@(s1-s0)/(2*step)
        assert np.allclose(Hv,H@v,rtol=2e-5,atol=2e-8)
        # Independent prior logpdf; penalty Hessian enters the same physical Jacobian.
        gp,hp=E.K.log_prior_grad_hess(e,model)
        def lp(ee):
            a,b,c,d=E.K.unpack(ee,model)
            return sum(PRIORS[q].logpdf(val).sum() for q,val in zip("abcd",(a,b,c,d)) if q!="d" or model=="4PL")
        assert abs((lp(e+step*v)-lp(e-step*v))/(2*step)-gp@v)<1e-6
        g1=E.K.log_prior_grad_hess(e+step*v,model)[0];g0=E.K.log_prior_grad_hess(e-step*v,model)[0]
        assert np.allclose((g1-g0)/(2*step),hp*v,rtol=2e-5,atol=1e-7)


@pytest.mark.parametrize("model",["3PL","4PL"])
def test_conditional_gradient_and_full_residual_functional(model):
    e=eta(model);px,s=E.terms(e,model);N=1e7
    # Construct an interior empirical distribution with exact penalized score zero.
    gp,_=E.K.log_prior_grad_hess(e,model);I=(px[:,None]*s).T@s
    phat=px*(1-s@np.linalg.solve(I,gp/N));phat/=phat.sum()
    assert phat.min()>0
    cond,dc=E.conditionals(e,model)
    assert np.allclose(cond,E.K.score_conditionals(e,model)[0],atol=1e-12)
    v=np.random.default_rng(9).normal(size=len(e));v/=np.linalg.norm(v)
    for step in (1e-5,5e-6):
        fd=(E.conditionals(e+step*v,model)[0]-E.conditionals(e-step*v,model)[0])/(2*step)
        assert np.allclose(fd,np.einsum("sjd,d->sj",dc,v),rtol=2e-5,atol=2e-8)
    _,score,A,_,_=E.bread(e,model,phat,N)
    perturb=px*np.cos(np.arange(len(px)));perturb-=px*perturb.sum()
    ks=[3,4];j=5;mask=np.isin(E.S,ks)
    h=mask*(E.X[:,j]-cond[E.S,j])
    pi=np.bincount(E.S,weights=phat,minlength=13)
    G=-(pi[ks,None]*dc[ks,j]).sum(0)
    analytic=perturb@(h+score@np.linalg.solve(A.T,G))
    def solve(w):
        def eq(ee):
            _,ss=E.terms(ee,model)
            return w@ss+E.K.log_prior_grad_hess(ee,model)[0]/N
        def jac(ee):
            _,_,hh=E.terms(ee,model,61,w)
            return hh+np.diag(E.K.log_prior_grad_hess(ee,model)[1])/N
        fit=optimize.root(eq,e,jac=jac,tol=1e-10)
        assert np.max(abs(eq(fit.x)))<1e-9
        cc=E.conditionals(fit.x,model)[0]
        return w@(mask*(E.X[:,j]-cc[E.S,j]))
    for step in (1e-5,5e-6):
        fd=(solve(phat+step*perturb)-solve(phat-step*perturb))/(2*step)
        assert np.isclose(fd,analytic,rtol=5e-4,atol=2e-8)


def test_centered_frequency_covariance_matches_expanded_rows():
    values=np.array([[1.,2.],[3.,-1.],[2.,5.]])
    counts=np.array([3,7,2]);expanded=np.repeat(values,counts,axis=0)
    assert np.allclose(E.covariance(values,counts/counts.sum()),np.cov(expanded,rowvar=False,bias=True))


def test_independent_positive_weight_tail_and_nullspace_shift():
    for lam,nc in ((np.ones(5),np.zeros(5)),(np.array([.2,.4,.9,1.1,1.3]),np.array([.2,-.3,.5,.1,0.]))):
        for q in (2.,7.,15.):
            sf,error=E.ruben_sf(q,lam,nc)
            assert error<1e-10
            assert abs(sf-E.K.imhof_sf(q,lam,nc))<1e-8
    p,rank,shift=E.qtail(7.,np.array([0.,1.,1.,1.]),np.array([2.,0.,0.,0.]))
    assert rank==3 and shift==4
    assert abs(p-stats.chi2.sf(3.,3))<1e-8
    with pytest.raises(ValueError):E.qtail(7.,np.array([-.1,1.,1.,1.]),np.zeros(4))


def test_primary_registry_and_original_gates_unchanged():
    import hashlib,json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    lock=json.loads((root/"validation/p3_empirical/PRESERVED.json").read_text())
    for path,sha in lock.items():
        raw=(root/path).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==sha
