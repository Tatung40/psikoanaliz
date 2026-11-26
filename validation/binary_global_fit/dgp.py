"""Fixed global-misfit DGP; original P3 evidence remains immutable."""
import numpy as np
from scipy.special import expit
from validation.binary_certification.common import truth
KINDS=("null","nonmonotone","local_dependence","second_factor")
def data(model,root,r,kind,N=3000):
    seed=[root,int(model[0]),r,KINDS.index(kind)]
    rng=np.random.default_rng(seed);theta=rng.standard_normal(N);u=rng.random((N,12))
    p=truth(model,12);a,b,c,d=p["a"],p["b"],p["c"],p["d"]
    z=a*(theta[:,None]-b)
    if kind=="second_factor":z[:,:3]+=.7*np.random.default_rng(seed+[9]).standard_normal(N)[:,None]
    P=c+(d-c)*expit(z);x=(u<P).astype(np.uint8)
    if kind=="nonmonotone":x[:,0]=u[:,0]<c[0]+(d[0]-c[0])*expit(z[:,0]-.25*(theta-b[0])**2)
    if kind=="local_dependence":x[:,0]=u[:,0]<c[0]+(d[0]-c[0])*expit(z[:,0]+(x[:,1]-P[:,1]))
    return x,seed
