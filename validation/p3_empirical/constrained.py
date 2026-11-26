"""Constrained-Gaussian estimating-equation residual reference; development only.
Physical MAP remains unchanged. This is not a bootstrap full-refit claim or ordinary chi-square.
"""
import time
import numpy as np
from validation.p3_empirical import kernel as E


def box_qp(A,Y,lower,upper,tol=1e-8,max_iter=100):
    """Batched exact active-set solution of .5*t'A*t-y't on a physical box.
    Independent scipy optimizations verify objectives and KKT in tests.
    """
    Y=np.asarray(Y,float);lo=np.broadcast_to(lower,Y.shape);hi=np.broadcast_to(upper,Y.shape)
    sol=np.linalg.solve(A,Y.T).T
    state=np.where(sol<lo,-1,np.where(sol>hi,1,0)).astype(np.int8)
    rows=np.arange(len(Y));worst=0.
    for iteration in range(max_iter):
        patterns,inv=np.unique(state,axis=0,return_inverse=True)
        for z,pattern in enumerate(patterns):
            ix=np.flatnonzero(inv==z);fixed=np.flatnonzero(pattern);free=np.flatnonzero(pattern==0)
            if len(fixed):sol[np.ix_(ix,fixed)]=np.where(pattern[fixed]<0,lo[np.ix_(ix,fixed)],hi[np.ix_(ix,fixed)])
            if len(free):
                rhs=Y[np.ix_(ix,free)]
                if len(fixed):rhs=rhs-sol[np.ix_(ix,fixed)]@A[np.ix_(fixed,free)]
                sol[np.ix_(ix,free)]=np.linalg.solve(A[np.ix_(free,free)],rhs.T).T
        violation=np.where(state==0,np.maximum(lo-sol,sol-hi),-np.inf)
        chosen=violation.argmax(1);add=violation[rows,chosen]>tol
        if add.any():
            rr=rows[add];cc=chosen[add];state[rr,cc]=np.where(sol[rr,cc]<lo[rr,cc],-1,1)
        gradient=sol@A-Y
        wrong=np.where(state==-1,-gradient,np.where(state==1,gradient,-np.inf))
        chosen=wrong.argmax(1);release=(wrong[rows,chosen]>tol)&~add
        if release.any():state[rows[release],chosen[release]]=0
        if not add.any() and not release.any():
            pg=np.where((sol<=lo+tol)&(gradient>0),0,np.where((sol>=hi-tol)&(gradient<0),0,gradient))
            worst=float(max(np.max(abs(pg)),np.max(lo-sol),np.max(sol-hi)))
            if worst>1e-7:raise ValueError("box QP KKT failed "+str(worst))
            return sol,{"iterations":iteration+1,"max_KKT":worst,"active_fraction":float(np.any(state!=0,axis=1).mean())}
    raise ValueError("box QP active-set did not converge")


def gaussian_reference(eta,model,N,H,G,px,score,seed,train=1024,B=4095):
    """Joint score/residual Gaussian law, prior drift and all native box constraints."""
    gp,hp=E.K.log_prior_grad_hess(eta,model)
    I=(px[:,None]*score).T@score
    A=I-np.diag(hp)/N
    if np.linalg.eigvalsh(A).min()<=0 or np.linalg.cond(A)>1e10:raise ValueError("invalid cone bread")
    m=px@H
    joint=np.c_[score,H-m];cov=E.covariance(joint,px)
    ev,Q=np.linalg.eigh((cov+cov.T)/2);scale=ev.max()
    if ev.min()<-1e-10*scale:raise ValueError("invalid joint Gaussian covariance")
    # Exact zero variance modes carry no random noise; their deterministic means remain.
    ev=np.maximum(ev,0)
    rng=np.random.default_rng(seed)
    draws=rng.standard_normal((train+B,len(ev)))@(Q*np.sqrt(ev)).T
    D=len(eta)
    ys=draws[:,:D]+gp/np.sqrt(N)
    kval=3 if model=="3PL" else 4
    bounds=np.array([E.BOUNDS[q] for j in range(12) for q in "abcd"[:kval]])
    lower=np.sqrt(N)*(bounds[:,0]-eta);upper=np.sqrt(N)*(bounds[:,1]-eta)
    step,diagnostics=box_qp(A,ys,lower,upper)
    residual=np.sqrt(N)*m+draws[:,D:]+step@G.T
    return residual,diagnostics


def analyse(x,eta,model,seed,train=1024,B=4095):
    start=time.perf_counter();x=np.asarray(x,int);N=len(x)
    phat=E.K.pattern_counts(x)/N
    px,score=E.terms(eta,model)
    cond,dc=E.conditionals(eta,model);pi=np.bincount(E.S,weights=px,minlength=13)
    nk=np.bincount(x.sum(1),minlength=13).astype(float)
    allH=[];allG=[];groups_all=[];weights=[];sxs=[];offsets=[0]
    for j in range(12):
        o1=np.bincount(x.sum(1),weights=x[:,j],minlength=13)
        O=np.c_[nk-o1,o1];EX=np.c_[nk*(1-cond[:,j]),nk*cond[:,j]]
        groups,sparse=E.K.pooling(O,EX);groups_all.append(groups)
        H=np.stack([np.isin(E.S,ks)*(E.X[:,j]-cond[E.S,j]) for ks in groups],1)
        G=-np.stack([(pi[ks,None]*dc[ks,j]).sum(0) for ks in groups])
        W=[];sx=0.
        for ks in groups:
            oo=O[ks].sum(0);ee=EX[ks].sum(0)
            sx+=float(np.sum((oo-ee)**2/ee));W.append((ee.sum()/N)/np.prod(ee/N))
        allH.append(H);allG.append(G);weights.append(np.sqrt(W));sxs.append(sx);offsets.append(offsets[-1]+len(groups))
    H=np.concatenate(allH,axis=1);G=np.concatenate(allG,axis=0)
    residual,diag=gaussian_reference(eta,model,N,H,G,px,score,seed,train,B)
    observed=np.sqrt(N)*(phat@H);items=[]
    for j in range(12):
        a,b=offsets[j:j+2];wh=weights[j]
        ref=residual[:,a:b]*wh;obs=observed[a:b]*wh
        raw=(ref[train:]**2).sum(1)
        p_raw=float((1+np.sum(raw>=sxs[j]))/(B+1))
        center=ref[:train].mean(0)
        Sig=np.cov(ref[:train],rowvar=False,ddof=1)
        lam,V=np.linalg.eigh((Sig+Sig.T)/2)
        if lam.min()<-1e-10*lam.max():raise ValueError("invalid standardized covariance")
        pos=lam>1e-10*lam.max();rank=int(pos.sum())
        if rank<3:raise ValueError("insufficient standardized rank")
        ov=V.T@(obs-center);rv=(ref[train:]-center)@V
        nullspace=float(np.linalg.norm(ov[~pos]))
        if nullspace>1e-6:
            q,pstd=float("nan"),float("nan")
        else:
            q=float(np.sum(ov[pos]**2/lam[pos]))
            sim=(rv[:,pos]**2/lam[pos]).sum(1)
            pstd=float((1+np.sum(sim>=q))/(B+1))
        items.append({"sx":sxs[j],"p_constrained":p_raw,"Q_constrained":q,"p_constrained_standardized":pstd,
            "rank":rank,"groups":groups_all[j],"ref_mean_norm":float(np.linalg.norm(center)),
            "min_eigenvalue":float(lam.min()),"max_eigenvalue":float(lam.max()),"nullspace_residual":nullspace})
    return {"items":items,"diagnostics":diag,"seconds":time.perf_counter()-start,"train":train,"B":B}
