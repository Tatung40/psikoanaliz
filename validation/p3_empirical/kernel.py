"""Validation-only empirical IF and separately named MAP-standardized residual Q.
No fitted parameters or production S-X2 definitions are changed.
"""
import time
import numpy as np
from scipy import stats
from scipy.special import expit, logsumexp
from validation.p3_sx2_analytic import kernel as K

X,S,J=K.X,K.S,K.J
BOUNDS={"a":(.1,5.),"b":(-6.,6.),"c":(0.,.5),"d":(.5,1.)}


def terms(eta,model,points=61,phat=None):
    """Louis identity in full physical coordinates. Hessian is per respondent."""
    t,w=K.quad(points); a,b,c,d=K.unpack(eta,model); k=4 if model=="4PL" else 3
    P,z=K.irf(eta,model,t)
    if np.any(P<=0) or np.any(P>=1):raise ValueError("noninterior probability")
    dp=K.irf_grad(eta,model,t); dz=z*(1-z); v=t[:,None]-b
    ll=X@np.log(P).T+(1-X)@np.log1p(-P).T
    logpx=logsumexp(ll+np.log(w),axis=1)
    px=np.exp(logpx); post=np.exp(ll+np.log(w)-logpx[:,None])
    fac=dp/(P*(1-P))[:,:,None]
    resid=X[:,None,:]-P[None,:,:]
    score=np.einsum("xq,xqj,qjk->xjk",post,resid,fac,optimize=True).reshape(len(X),-1)
    if phat is None:return px,score
    mass=np.asarray(phat)[:,None]*post
    nq=mass.sum(0); mx=mass.T@X
    rmean=mx-nq[:,None]*P
    second=np.empty((len(t),J,J))
    for q in range(len(t)):
        second[q]=X.T@(mass[:,q,None]*X)-np.outer(mx[q],P[q])-np.outer(P[q],mx[q])+nq[q]*np.outer(P[q],P[q])
    outer=np.einsum("qij,qia,qjb->iajb",second,fac,fac,optimize=True).reshape(J*k,J*k)
    H=outer-(np.asarray(phat)[:,None]*score).T@score
    dd=np.zeros((len(t),J,k,k))
    dd[:,:,0,0]=(d-c)*dz*(1-2*z)*v*v
    dd[:,:,0,1]=dd[:,:,1,0]=-(d-c)*dz*(1+a*v*(1-2*z))
    dd[:,:,1,1]=(d-c)*dz*(1-2*z)*a*a
    dd[:,:,0,2]=dd[:,:,2,0]=-dz*v
    dd[:,:,1,2]=dd[:,:,2,1]=a*dz
    if k==4:
        dd[:,:,0,3]=dd[:,:,3,0]=dz*v
        dd[:,:,1,3]=dd[:,:,3,1]=-a*dz
    hmean=rmean/(P*(1-P))
    hsecond=-mx/P**2-(nq[:,None]-mx)/(1-P)**2
    blocks=np.einsum("qj,qjab->jab",hmean,dd)+np.einsum("qj,qja,qjb->jab",hsecond,dp,dp)
    for j in range(J):H[j*k:(j+1)*k,j*k:(j+1)*k]+=blocks[j]
    return px,score,(H+H.T)/2


def conditionals(eta,model):
    """Analytic derivative of the native GH41 raw-score conditional probabilities."""
    px,score=terms(eta,model,41); D=len(eta)
    marg=np.bincount(S,weights=px,minlength=13)
    dm=np.zeros((13,D)); joint=np.zeros((13,J)); dj=np.zeros((13,J,D))
    for s in range(13):
        ix=S==s; w=px[ix]
        dm[s]=w@score[ix];joint[s]=w@X[ix]
        dj[s]=(w[:,None]*X[ix]).T@score[ix]
    cond=joint/marg[:,None]
    deriv=(dj-cond[:,:,None]*dm[:,None,:])/marg[:,None,None]
    return cond,deriv


def bread(eta,model,phat,N,expected=False):
    px,score,H=terms(eta,model,61,phat)
    gp,hp=K.log_prior_grad_hess(eta,model)
    I=(px[:,None]*score).T@score
    A=(I if expected else -H)-np.diag(hp)/N
    A=(A+A.T)/2
    eig=np.linalg.eigvalsh(A)
    if eig[0]<=0 or np.linalg.cond(A)>1e10:raise ValueError("invalid full penalized bread")
    return px,score,A,gp,{"min_bread":float(eig[0]),"bread_condition":float(np.linalg.cond(A)),
        "expected_observed_relative":float(np.linalg.norm(-H-I)/np.linalg.norm(I)),
        "max_MAP_score":float(np.max(abs(phat@score+gp/N)))}


def covariance(psi,weights):
    mean=weights@psi
    centered=psi-mean
    return (weights[:,None]*centered).T@centered


def qtail(q,lam,mean):
    """Generalized chi-square, retaining deterministic zero-eigenvalue means."""
    lam=np.asarray(lam);mean=np.asarray(mean)
    scale=max(float(lam.max()),1e-300)
    if lam.min() < -1e-10*scale:raise ValueError("negative residual eigenvalue")
    pos=lam>1e-10*scale
    if pos.sum()<3:raise ValueError("residual rank below3")
    shift=float(np.sum(mean[~pos]**2)); threshold=q-shift
    if threshold<=0:return 1.,int(pos.sum()),shift
    nc=mean[pos]/np.sqrt(lam[pos])
    return K.imhof_sf(threshold,lam[pos],nc),int(pos.sum()),shift


def ruben_sf(q,lam,nc=None,tol=1e-11):
    """Independent positive-weight gamma-mixture series, for tail verification only."""
    lam=np.asarray(lam,float); nc=np.zeros_like(lam) if nc is None else np.asarray(nc)
    beta=float(lam.min());rho=1-beta/lam; n=len(lam)
    ak=[float(np.exp(.5*np.log(beta/lam).sum()-.5*(nc*nc).sum()))]
    ck=[];total=ak[0];sf=ak[0]*stats.chi2.sf(q/beta,n)
    for k in range(1,20000):
        ck.append(float(.5*np.sum(rho**k)+.5*k*np.sum(nc*nc*(1-rho)*rho**(k-1))))
        val=float(np.dot(ck,ak[::-1])/k);ak.append(val);total+=val
        sf+=val*stats.chi2.sf(q/beta,n+2*k)
        if 1-total<tol:return float(sf),float(max(0,1-total))
    raise ValueError("independent tail series did not converge")


def analyse(x,eta,model,expected=False):
    start=time.perf_counter();x=np.asarray(x,int);N=len(x)
    phat=K.pattern_counts(x)/N
    k=4 if model=="4PL" else 3;e=np.asarray(eta).reshape(12,k)
    for j,q in enumerate("abcd"[:k]):
        lo,hi=BOUNDS[q]
        if np.any(e[:,j]-lo<=1e-6) or np.any(hi-e[:,j]<=1e-6):raise ValueError("active native parameter bound")
    px,score,A,gp,diag=bread(eta,model,phat,N,expected)
    cond,dcond=conditionals(eta,model)
    nk=np.bincount(x.sum(1),minlength=13).astype(float)
    empirical_pi=np.bincount(S,weights=phat,minlength=13)
    output=[]
    for j in range(12):
        o1=np.bincount(x.sum(1),weights=x[:,j],minlength=13)
        O=np.c_[nk-o1,o1];E=np.c_[nk*(1-cond[:,j]),nk*cond[:,j]]
        groups,sparse=K.pooling(O,E)
        H=np.stack([np.isin(S,ks)*(X[:,j]-cond[S,j]) for ks in groups],axis=1)
        G=-np.stack([(empirical_pi[ks,None]*dcond[ks,j]).sum(0) for ks in groups])
        GA=np.linalg.solve(A.T,G.T).T
        psi=H+score@GA.T
        Sig=covariance(psi,phat)
        u=phat@H
        # Model raw residual mean includes GH61/GH41 discrepancy. Prior mean
        # is first-order, not asserted exact: size and power require fresh MC.
        nullmean=px@H+GA@gp/N
        W=[];sx=0.
        for ks in groups:
            o=O[ks].sum(0);ee=E[ks].sum(0)
            sx+=float(np.sum((o-ee)**2/ee))
            W.append(float((ee.sum()/N)/np.prod(ee/N)))
        wh=np.sqrt(W);M=wh[:,None]*Sig*wh[None]
        lam,V=np.linalg.eigh((M+M.T)/2)
        mean=V.T@(np.sqrt(N)*wh*nullmean)
        p,rank,shift=qtail(sx,lam,mean)
        centered=V.T@(np.sqrt(N)*wh*(u-nullmean))
        pos=lam>1e-10*max(float(lam.max()),1e-300)
        null_resid=float(np.linalg.norm(centered[~pos]))
        if null_resid>1e-6:
            qs,ps=float("nan"),float("nan")
        else:
            qs=float(np.sum(centered[pos]**2/lam[pos]));ps=float(stats.chi2.sf(qs,rank))
        output.append(dict(sx=sx,sx_from_u=float(N*np.sum(np.asarray(W)*u*u)),
            p_emp=float(p),standardized_Q=qs,p_standardized=ps,rank=rank,groups=groups,sparse=sparse,
            min_eigenvalue=float(lam.min()),max_eigenvalue=float(lam.max()),
            deterministic_shift=shift,nullspace_residual=null_resid,
            residual_model_mean=float(np.linalg.norm(px@H)),prior_mean=float(np.linalg.norm(GA@gp/N)),
            variance_method="empirical-centered / observed-joint-penalized"))
    return {"items":output,"diagnostics":diag,"seconds":time.perf_counter()-start}
