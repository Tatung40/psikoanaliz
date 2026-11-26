"""Observable KL lower bound for ANY locally independent monotone positive-slope IRT null.
For rest-score T, P(Xj=1|T=t) is nondecreasing:
successive Poisson-binomial probabilities have ratio e_(t+1)(odds)/e_t(odds),
coordinatewise increasing by Newton's inequalities. All rest-item odds increase
with theta. Thus the kernel(theta,T) is TP2, theta|T is MLR ordered, and
the increasing target IRF has increasing conditional expectation.
Data processing + Bernoulli KL isotonic projection gives a GLOBAL lower bound
on KL(Q || any admissible positive-slope 4PL marginal law), not a local optimum.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.special import xlogy
from validation.p3_empirical import kernel as E
from validation.p3_4pl_closure.dgp import distribution

def pav(values,weights):
    blocks=[]
    for i,(v,w) in enumerate(zip(values,weights)):
        blocks.append([i,i+1,float(w),float(w*v)])
        while len(blocks)>1 and blocks[-2][3]/blocks[-2][2]>blocks[-1][3]/blocks[-1][2]:
            b=blocks.pop();a=blocks.pop();blocks.append([a[0],b[1],a[2]+b[2],a[3]+b[3]])
    out=np.empty(len(values))
    for a,b,w,s in blocks:out[a:b]=s/w
    return out

def bound(q,j=0):
    t=(E.S-E.X[:,j]).astype(np.int64)
    mass=np.bincount(t,weights=q,minlength=12)
    rate=np.bincount(t,weights=q*E.X[:,j],minlength=12)/mass
    null=pav(rate,mass)
    kl=float(mass@(xlogy(rate,rate/null)+xlogy(1-rate,(1-rate)/(1-null))))
    return dict(rest_score_probability=mass.tolist(),conditional_item_probability=rate.tolist(),
        closest_monotone_relaxation=null.tolist(),KL_global_lower=max(0.,kl),N_KL_global_lower=3000*max(0.,kl))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);z=ap.parse_args()
    out={}
    for kind in ("null","nonmonotone","local_dependence","two_stage"):
        q=distribution(kind,241);q2=distribution(kind,121)
        out[kind]=bound(q);out[kind]["GH121_KL_global_lower"]=bound(q2)["KL_global_lower"]
    Path(z.out).parent.mkdir(parents=True,exist_ok=True);Path(z.out).write_text(json.dumps(out,indent=1))
    print("WITNESS_JSON "+json.dumps(out),flush=True)
if __name__=="__main__":main()
