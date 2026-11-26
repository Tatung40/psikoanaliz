"""Separately named orthogonal marginal residual score; MAP plug-in is not assumed ML."""
import time
import numpy as np
from scipy import stats
from validation.p3_empirical import kernel as E


def efficient_basis(px,score,H):
    centered=H-px@H
    I=(px[:,None]*score).T@score
    if np.linalg.eigvalsh(I).min()<=0 or np.linalg.cond(I)>1e12:raise ValueError("invalid nuisance likelihood information")
    C=score.T@(px[:,None]*centered)
    T=centered-score@np.linalg.solve(I,C)
    orth=float(np.max(abs(score.T@(px[:,None]*T))))
    if orth>1e-8:raise ValueError("efficient score not nuisance orthogonal")
    return T,E.covariance(T,px),orth


def analyse(x,eta,model):
    start=time.perf_counter();x=np.asarray(x,int);N=len(x);phat=E.K.pattern_counts(x)/N
    px,score=E.terms(eta,model);cond,dc=E.conditionals(eta,model)
    nk=np.bincount(x.sum(1),minlength=13).astype(float);items=[]
    for j in range(12):
        o1=np.bincount(x.sum(1),weights=x[:,j],minlength=13)
        O=np.c_[nk-o1,o1];EX=np.c_[nk*(1-cond[:,j]),nk*cond[:,j]]
        groups,_=E.K.pooling(O,EX)
        H=np.stack([np.isin(E.S,ks)*(E.X[:,j]-cond[E.S,j]) for ks in groups],1)
        T,Sig,orth=efficient_basis(px,score,H)
        ev,Q=np.linalg.eigh((Sig+Sig.T)/2);pos=ev>1e-10*ev.max();rank=int(pos.sum())
        if ev.min()<-1e-10*ev.max() or rank<3:raise ValueError("invalid efficient rank")
        u=Q.T@(np.sqrt(N)*(phat@T))
        if np.linalg.norm(u[~pos])>1e-6:q,p=float("nan"),float("nan")
        else:q=float(np.sum(u[pos]**2/ev[pos]));p=float(stats.chi2.sf(q,rank))
        sx=sum(float(np.sum((O[ks].sum(0)-EX[ks].sum(0))**2/EX[ks].sum(0))) for ks in groups)
        items.append({"sx":sx,"Q_efficient":q,"p_efficient":p,"rank":rank,"orthogonality":orth,
            "min_eigenvalue":float(ev.min()),"max_eigenvalue":float(ev.max()),"groups":groups})
    return {"items":items,"seconds":time.perf_counter()-start}
