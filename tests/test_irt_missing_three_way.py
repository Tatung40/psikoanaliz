import numpy as np
from scipy.special import expit, logsumexp
from numpy.polynomial.hermite import hermgauss
from core.irt_missing_likelihood import marginal_loglik_missing

def test_three_independent_missing_loglik_implementations():
    x=np.array([[1.,np.nan,0.,1.],[0.,1.,np.nan,0.],[np.nan,1.,1.,1.]])
    a=np.array([.7,1.1,1.4,.9]);b=np.array([-.6,.2,.8,-.2])
    c=np.array([.1,.05,.15,.08]);d=np.array([.95,.99,.9,.96])
    points=17
    platform=marginal_loglik_missing(x,a,b,c,d,points=points)
    nodes,weights=hermgauss(points)
    theta=np.sqrt(2)*nodes
    p=c[None,:]+(d-c)[None,:]*expit(a[None,:]*(theta[:,None]-b[None,:]))
    # External SciPy logsumexp with masked vectorization
    terms=[]
    for row in x:
        valid=np.isfinite(row)
        lp=np.where(row[valid][None,:]==1,np.log(p[:,valid]),np.log1p(-p[:,valid])).sum(axis=1)
        terms.append(logsumexp(lp+np.log(weights/np.sqrt(np.pi))))
    external=float(sum(terms))
    # Independent scalar quadrature using math only
    import math
    reference=0.
    for row in x:
        mixture=0.
        for t,w in zip(theta,weights):
            response=1.
            for j,value in enumerate(row):
                if math.isnan(value):
                    continue
                logistic=1/(1+math.exp(-float(a[j])*(float(t)-float(b[j]))))
                prob=float(c[j])+(float(d[j])-float(c[j]))*logistic
                response*=prob if value==1 else (1-prob)
            mixture+=float(w)*response/math.sqrt(math.pi)
        reference+=math.log(mixture)
    assert abs(platform-external)<1e-10
    assert abs(platform-reference)<1e-10
