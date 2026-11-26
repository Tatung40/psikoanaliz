"""Constrained projection verified against independent scipy convex optimization."""
import numpy as np
from scipy import optimize
from validation.p3_empirical.constrained import box_qp,gaussian_reference
from validation.p3_empirical import kernel as E


def test_active_set_matches_independent_optimizer_and_unconstrained_limit():
    rng=np.random.default_rng(371);D=8;M=rng.normal(size=(D,D));A=M.T@M+np.eye(D)*.2
    Y=rng.normal(size=(64,D));lo=np.full(D,-.15);hi=np.full(D,.2)
    sol,diag=box_qp(A,Y,lo,hi)
    assert diag["max_KKT"]<1e-7
    for i in range(len(Y)):
        y=Y[i]
        r=optimize.minimize(lambda x:(.5*x@A@x-y@x,A@x-y),np.zeros(D),jac=True,
            method="L-BFGS-B",bounds=list(zip(lo,hi)),options={"gtol":1e-11,"ftol":1e-15,"maxiter":5000})
        assert np.max(abs(r.x-sol[i]))<1e-6
        assert abs(r.fun-(.5*sol[i]@A@sol[i]-y@sol[i]))<1e-10
    free,_=box_qp(A,Y,np.full(D,-1e9),np.full(D,1e9))
    assert np.allclose(free,np.linalg.solve(A,Y.T).T,atol=1e-12)


def test_joint_gaussian_unconstrained_covariance_and_bias():
    from tests.test_p3_empirical import eta
    e=eta("3PL");N=1e9;px,score=E.terms(e,"3PL");cond,dc=E.conditionals(e,"3PL")
    pi=np.bincount(E.S,weights=px,minlength=13)
    H=np.stack([(E.S==k)*(E.X[:,0]-cond[E.S,0]) for k in (3,4,5)],1)
    G=np.stack([-pi[k]*dc[k,0] for k in (3,4,5)])
    gp,hp=E.K.log_prior_grad_hess(e,"3PL");I=(px[:,None]*score).T@score;A=I-np.diag(hp)/N
    psi=H+score@np.linalg.solve(A.T,G.T)
    target=E.covariance(psi,px)
    ref,diag=gaussian_reference(e,"3PL",N,H,G,px,score,[99,1],train=1024,B=4095)
    assert diag["active_fraction"]==0
    assert np.max(abs(np.cov(ref,rowvar=False)-target))<.003
    mean=np.sqrt(N)*(px@H)+G@np.linalg.solve(A,gp)/np.sqrt(N)
    assert np.max(abs(ref.mean(0)-mean))<.03


def test_efficient_score_projection_and_local_orthogonality():
    from tests.test_p3_empirical import eta
    from validation.p3_empirical.efficient import efficient_basis
    e=eta("3PL");px,score=E.terms(e,"3PL")
    def basis(ee):
        pp,ss=E.terms(ee,"3PL");cc=E.conditionals(ee,"3PL")[0]
        H=np.stack([(E.S==k)*(E.X[:,0]-cc[E.S,0]) for k in (3,4,5,6)],1)
        return efficient_basis(pp,ss,H)
    T,Sig,error=basis(e)
    assert np.max(abs(px@T))<1e-9 and error<1e-8
    assert np.allclose(Sig,E.covariance(T,px),atol=1e-12)
    v=np.random.default_rng(11).normal(size=len(e));v/=np.linalg.norm(v)
    for step in (1e-5,5e-6):
        fd=px@(basis(e+step*v)[0]-basis(e-step*v)[0])/(2*step)
        assert np.max(abs(fd))<1e-6
