"""Constraint-aware candidate references for the UNCHANGED public native M2, and separate QLI."""
import time
import numpy as np
from scipy import linalg,stats
from validation.p3_empirical import kernel as E
from validation.p3_empirical.constrained import box_qp
from validation.binary_global_fit import kernel as K

def nonlinear_means(eta,steps,model,N,points=61,batch=256):
    J=12;k=3 if model=="3PL" else 4;t,w=K.grid(points)
    values=[];sets=K.structure(J)[0]
    for start in range(0,len(steps),batch):
        par=(eta+steps[start:start+batch]/np.sqrt(N)).reshape(-1,J,k)
        a,b,c=par[:,:,0],par[:,:,1],par[:,:,2];d=par[:,:,3] if k==4 else np.ones_like(c)
        from scipy.special import expit
        P=c[:,None,:]+(d-c)[:,None,:]*expit(a[:,None,:]*(t[None,:,None]-b[:,None,:]))
        uni=np.einsum("bqj,q->bj",P,w)
        weighted=P*np.sqrt(w)[None,:,None]
        cross=weighted.transpose(0,2,1)@weighted
        left,right=np.triu_indices(J,1)
        mu=np.c_[uni,cross[:,left,right]]
        values.append(mu)
    return np.concatenate(values)

def analyse(x,eta,model,seed,train=1024,B=4095,nonlinear=True):
    start=time.perf_counter();x=np.asarray(x,int);N,J=x.shape
    mu,Xi,D=K.moments(eta,model,J)
    obs=K.observed(x)
    m2,diag,L,U=K.m2_from_moments(obs,mu,Xi,D,N,rank_threshold=1e-6)
    px,score=E.terms(eta,model)
    Z=np.stack([np.prod(E.X[:,S],axis=1) for S in K.structure(J)[0]],axis=1)
    H=Z-mu
    gp,hp=E.K.log_prior_grad_hess(eta,model)
    I=(px[:,None]*score).T@score;A=I-np.diag(hp)/N
    if np.linalg.eigvalsh(A).min()<=0 or np.linalg.cond(A)>1e10:raise ValueError("Unsafe full MAP bread")
    joint=np.c_[score,H-px@H];cov=E.covariance(joint,px)
    ev,V=np.linalg.eigh((cov+cov.T)/2)
    if ev.min()<-1e-10*ev.max():raise ValueError("Invalid joint covariance")
    rng=np.random.default_rng(seed);noise=rng.standard_normal((train+B,len(ev)))@(V*np.sqrt(np.maximum(ev,0))).T
    k=3 if model=="3PL" else 4;q=len(eta);bounds=np.array([E.BOUNDS[v] for j in range(J) for v in "abcd"[:k]])
    step,kkt=box_qp(A,noise[:,:q]+gp/np.sqrt(N),np.sqrt(N)*(bounds[:,0]-eta),np.sqrt(N)*(bounds[:,1]-eta))
    raw=np.sqrt(N)*(px@H)+noise[:,q:]
    linear=raw-step@D.T
    ref=raw-np.sqrt(N)*(nonlinear_means(eta,step,model,N)-mu) if nonlinear else linear
    projector=linalg.solve_triangular(L,U,lower=True,trans="T").T
    # projector @ residual is the actual native M2 norm; rank rule is explicitly1e-6.
    mlin=np.sum((linear[train:]@projector.T)**2,axis=1)
    mnon=np.sum((ref[train:]@projector.T)**2,axis=1)
    observed=np.sqrt(N)*(obs-mu)
    p_lin=float((1+(mlin>=m2).sum())/(B+1))
    p_non=float((1+(mnon>=m2).sum())/(B+1))
    center=ref[:train].mean(0);S=np.cov(ref[:train],rowvar=False)
    lam,V=np.linalg.eigh((S+S.T)/2)
    if lam.min()<-1e-10*lam.max():raise ValueError("Negative residual covariance")
    ranks=[int((lam>cut*lam.max()).sum()) for cut in (1e-9,1e-10,1e-11)]
    pos=lam>1e-10*lam.max();z=V.T@(observed-center);nullspace=float(np.linalg.norm(z[~pos]))
    if min(ranks)<3 or len(set(ranks))!=1 or nullspace>1e-6:
        Q,pQ=float("nan"),float("nan")
    else:
        Q=float(np.sum(z[pos]**2/lam[pos]))
        sim=(ref[train:]-center)@V
        stat=np.sum(sim[:,pos]**2/lam[pos],axis=1)
        pQ=float((1+(stat>=Q).sum())/(B+1))
    return dict(M2=m2,df=diag["df"],p_chi=float(stats.chi2.sf(m2,diag["df"])),
        p_linear=p_lin,p_nonlinear=p_non,QLI=Q,p_QLI=pQ,rank=diag,QLI_rank=ranks,
        nullspace_residual=nullspace,max_KKT=kkt["max_KKT"],active_fraction=kkt["active_fraction"],
        seconds=time.perf_counter()-start,mean_norm=float(np.linalg.norm(center)))
