"""Stable-quadrature and feasible-null diagnostics only; NO new optimization."""
import argparse,json,math
from pathlib import Path
import numpy as np
from scipy.special import expit,logsumexp,log_expit
from validation.binary_certification.common import truth
from validation.p3_empirical.population import alternative
from validation.p3_empirical import kernel as E


def feasible_constant_item(model,marginal,points):
    p=truth(model,12);p={k:np.array(v,copy=True) for k,v in p.items()}
    t,w=E.K.quad(points)
    if model=="4PL":
        p["a"][0]=5.;p["b"][0]=-6.
        avg=float(w@expit(5*(t+6)))
        p["d"][0]=marginal  # explicit feasible null; no numerical matching needed
        assert .5<=p["d"][0]<=1.
    else:
        p["a"][0]=.1;p["b"][0]=-6.
        avg=float(w@expit(.1*(t+6)))
        p["c"][0]=np.clip((marginal-avg)/(1-avg),0,.5)
    z=p["a"]*(t[:,None]-p["b"])
    with np.errstate(divide="ignore"):
        lp=np.logaddexp(np.log(p["c"]),np.log(p["d"]-p["c"])+log_expit(z))
        lq=np.logaddexp(np.log1p(-p["d"]),np.log(p["d"]-p["c"])+log_expit(-z))
    logp=logsumexp(E.X@lp.T+(1-E.X)@lq.T+np.log(w),axis=1)
    return np.exp(logp),{k:float(v[0]) for k,v in p.items()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);z=ap.parse_args();out={}
    for model in ("3PL","4PL"):
        q,m=alternative(model,241);q2,_=alternative(model,121)
        p,coef=feasible_constant_item(model,m,241);p2,_=feasible_constant_item(model,m,121)
        KL=float(q@np.log(q/p));KL2=float(q2@np.log(q2/p2))
        assert np.isfinite(KL) and KL>=-1e-12
        bound=min(1.,.05+np.sqrt(3000*max(KL,0)/2))
        out[model]={"feasible_item1":coef,"alternative_marginal":m,"alternative_GH121_GH241_L1":float(abs(q-q2).sum()),
            "feasible_null_GH121_GH241_L1":float(abs(p-p2).sum()),"KL_GH241":KL,"KL_GH121":KL2,
            "N_KL":3000*KL,"uniform_level_.05_power_upper_Pinsker":bound,
            "analytic_coupling_power_bound":(.05+3000*.8*math.exp(-17.5)) if model=="4PL" else None,
            "analytic_derivation":("Take a1=5,b1=-6,c1=.1,d1=m, where m=E[.1+.8 logistic(1.6(theta+1.25))] lies in(.5,.9). Couple same theta, other items and uniform first-item responses. The per-person disagreement is E[(m-.1) logistic(-5(theta+6))] <=.8 E[exp(-5theta-30)]=.8 exp(-17.5). For N independent persons TV<=N times this bound; any test of size<=.05 at this feasible null has alternative power<=.05+N*.8 exp(-17.5). This is a uniform composite-null statement, not a claim that the single registered reference cell proves uniform level.") if model=="4PL" else None,
            "interpretation":"Bound applies to any test controlling .05 at this feasible null, not merely at the one registered reference truth. It does not prove structural parameter nonidentifiability or a global optimum. Native MAP prior is unchanged; this is an explicit feasible member of the model family."}
    Path(z.out).parent.mkdir(parents=True,exist_ok=True);Path(z.out).write_text(json.dumps(out,indent=1))
    print("FEASIBLE_NULL_DIAG_JSON "+json.dumps(out),flush=True)


if __name__=="__main__":main()
