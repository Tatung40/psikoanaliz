"""Prospectively fixed 4PL departures; no historical generator is altered."""
import numpy as np
from scipy.special import expit,logsumexp
from validation.binary_certification.common import truth
from validation.p3_empirical import kernel as E
KINDS=("null","nonmonotone","local_dependence","two_stage")

def first_probability(theta,kind,x2=None):
    p=truth("4PL",12);z=p["a"][0]*(theta-p["b"][0])
    if kind=="nonmonotone":z=z-.25*(theta-p["b"][0])**2
    if kind=="local_dependence":
        p2=p["c"][1]+(p["d"][1]-p["c"][1])*expit(p["a"][1]*(theta-p["b"][1]))
        z=z+1.0*(np.asarray(x2)-p2)
    f=.6*expit(2*(theta+1))+.4*expit(3*(theta-1)) if kind=="two_stage" else expit(z)
    return p["c"][0]+(p["d"][0]-p["c"][0])*f

def data(root,r,kind,N=3000):
    seed=[int(root),4,int(r),KINDS.index(kind)]
    rng=np.random.default_rng(seed);theta=rng.standard_normal(N);u=rng.random((N,12))
    p=truth("4PL",12)
    P=p["c"]+(p["d"]-p["c"])*expit(p["a"]*(theta[:,None]-p["b"]))
    x=(u<P).astype(np.uint8)
    if kind!="null":x[:,0]=u[:,0]<first_probability(theta,kind,x[:,1])
    return x,seed

def distribution(kind,points=121):
    theta,w=E.K.quad(points);p=truth("4PL",12)
    P=p["c"]+(p["d"]-p["c"])*expit(p["a"]*(theta[:,None]-p["b"]))
    logL=E.X@np.log(P).T+(1-E.X)@np.log1p(-P).T
    if kind!="null":
        for x2 in (0,1):
            ix=E.X[:,1]==x2
            z=first_probability(theta,kind,x2)
            logL[ix]+=E.X[ix,0,None]*(np.log(z)-np.log(P[:,0]))+(1-E.X[ix,0,None])*(np.log1p(-z)-np.log1p(-P[:,0]))
    q=np.exp(logsumexp(logL+np.log(w),axis=1))
    assert abs(q.sum()-1)<1e-12
    return q

def truth_eta():
    p=truth("4PL",12)
    return np.stack([p[k] for k in "abcd"],axis=1).ravel()
