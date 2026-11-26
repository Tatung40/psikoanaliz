"""Limited-information binary moments and rank-aware GLS diagnostics; no estimator."""
from functools import lru_cache
from itertools import combinations
import numpy as np
from scipy import linalg
from scipy.special import expit,roots_hermite
from scipy.stats import norm

@lru_cache(None)
def structure(J):
    if not 3<=J<=30:raise ValueError("Binary global moments support3..30 items.")
    sets=[(j,) for j in range(J)]+list(combinations(range(J),2))
    masks=np.array([sum(1<<j for j in s) for s in sets],dtype=np.int64)
    unions,inverse=np.unique((masks[:,None]|masks[None,:]).ravel(),return_inverse=True)
    membership=((unions[:,None]>>np.arange(J))&1).astype(bool)
    return sets,membership,inverse

def grid(points=61,kind="GH"):
    if kind=="GH":
        z,w=roots_hermite(points);return np.sqrt(2)*z,w/np.sqrt(np.pi)
    if kind=="mirt":
        z=np.linspace(-6,6,points);w=norm.pdf(z);return z,w/w.sum()
    raise ValueError("Unsupported integration rule.")

def unpack(eta,model,J):
    if model in ("Rasch","fixed1PL"):return np.ones(J),np.array(eta),np.zeros(J),np.ones(J)
    if model=="common1PL":return np.full(J,eta[-1]),np.array(eta[:-1]),np.zeros(J),np.ones(J)
    k={"2PL":2,"3PL":3,"4PL":4}[model];v=np.asarray(eta).reshape(J,k)
    return v[:,0],v[:,1],v[:,2] if k>=3 else np.zeros(J),v[:,3] if k==4 else np.ones(J)

def moment_core(P,w):
    Q,J=P.shape;sets,membership,inverse=structure(J)
    means=np.array([w@np.prod(P[:,s],axis=1) for s in sets])
    unions=np.ones((len(membership),Q))
    for j in range(J):
        take=membership[:,j];unions[take]*=P[:,j]
    expectations=unions@w
    cov=expectations[inverse].reshape(len(sets),len(sets))-np.outer(means,means)
    return means,(cov+cov.T)/2

def moments(eta,model,J,points=61,kind="GH",jacobian=True):
    t,w=grid(points,kind);a,b,c,d=unpack(np.asarray(eta,float),model,J)
    if not all(np.isfinite(v).all() for v in (a,b,c,d)):raise ValueError("Nonfinite parameters.")
    if np.any(c<0)|np.any(d>1)|np.any(c>=d)|np.any(a<=0):raise ValueError("Invalid physical IRF parameters.")
    L=expit(a*(t[:,None]-b));P=c+(d-c)*L
    mu,Xi=moment_core(P,w)
    if not jacobian:return mu,Xi
    sets,_,_=structure(J)
    dp=np.stack(((d-c)*L*(1-L)*(t[:,None]-b),-(d-c)*L*(1-L)*a,1-L,L),axis=2)
    physical=np.zeros((len(sets),J,4))
    for s,S in enumerate(sets):
        for j in S:
            other=[k for k in S if k!=j]
            factor=np.prod(P[:,other],axis=1) if other else np.ones(len(t))
            physical[s,j]=np.einsum("q,qk->k",w*factor,dp[:,j])
    if model in ("Rasch","fixed1PL"):D=physical[:,:,1]
    elif model=="common1PL":D=np.c_[physical[:,:,1],physical[:,:,0].sum(1)]
    else:D=physical[:,:,:{"2PL":2,"3PL":3,"4PL":4}[model]].reshape(len(sets),-1)
    return mu,Xi,D

def observed(x):
    x=np.asarray(x,float)
    if x.ndim!=2 or not np.isfinite(x).all() or not np.isin(x,[0,1]).all():raise ValueError("Complete binary data required; no imputation/deletion.")
    sets,_,_=structure(x.shape[1])
    return np.array([np.prod(x[:,S],axis=1).mean() for S in sets])

def projection(Xi,D,rank_threshold=1e-10):
    ev=np.linalg.eigvalsh(Xi)
    if ev[0]<=0 or ev[-1]/ev[0]>1e10:raise ValueError("Moment covariance not safely positive definite.")
    L=linalg.cholesky(Xi,lower=True);W=linalg.solve_triangular(L,D,lower=True)
    norms=np.linalg.norm(W,axis=0)
    if np.any(norms==0):raise ValueError("Zero moment-Jacobian column.")
    U,s,V=linalg.svd(W/norms,full_matrices=True)
    ranks=[int(np.sum(s>v*s[0])) for v in (rank_threshold*10,rank_threshold,rank_threshold/10)]
    rank=ranks[1];basis=U[:,rank:];df=len(Xi)-rank
    if df<=0:raise ValueError("No residual degrees of freedom.")
    return L,basis,dict(rank=rank,rank_stable=len(set(ranks))==1,rank_audit=ranks,df=df,
        q_nominal=D.shape[1],Xi_condition=float(ev[-1]/ev[0]),jacobian_singular_values=s.tolist())

def m2_from_moments(obs,mu,Xi,D,N,rank_threshold=1e-10):
    L,U,diag=projection(Xi,D,rank_threshold)
    v=linalg.solve_triangular(L,np.asarray(obs)-mu,lower=True)
    statistic=float(N*np.sum((U.T@v)**2))
    return statistic,diag,L,U

def baseline(obs,J,N):
    p=np.asarray(obs[:J]);sets,_,_=structure(J)
    if np.any(p<=0)|np.any(p>=1):raise ValueError("Independence baseline at a data boundary.")
    mu,Xi=moment_core(p[None,:],np.ones(1));D=np.zeros((len(sets),J))
    for i,S in enumerate(sets):
        for j in S:D[i,j]=np.prod(p[[k for k in S if k!=j]])
    stat,diag,_,_=m2_from_moments(obs,mu,Xi,D,N)
    return stat,diag

def srmsr(obs,mu,J):
    pairs=list(combinations(range(J),2));po=np.asarray(obs[:J]);pm=np.asarray(mu[:J])
    if np.any(po*(1-po)<=0)|np.any(pm*(1-pm)<=0):raise ValueError("Undefined observed/model correlation.")
    residual=np.eye(J)*0.
    for z,(i,j) in enumerate(pairs):
        ro=(obs[J+z]-po[i]*po[j])/np.sqrt(po[i]*(1-po[i])*po[j]*(1-po[j]))
        rm=(mu[J+z]-pm[i]*pm[j])/np.sqrt(pm[i]*(1-pm[i])*pm[j]*(1-pm[j]))
        residual[i,j]=residual[j,i]=ro-rm
    return float(np.sqrt(np.mean(residual[np.triu_indices(J,1)]**2))),residual
