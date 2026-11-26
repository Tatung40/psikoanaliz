"""Deterministic diagnostics, not new data or proof of global KL optimum."""
import argparse,json,time,math
from pathlib import Path
import numpy as np
from scipy import optimize,integrate,stats
from scipy.special import expit
from validation.p3_empirical import kernel as E
from validation.p3_4pl_closure.dgp import distribution,truth_eta,first_probability

def independent_old_bound():
    m=integrate.quad(lambda t:(.1+.8*expit(1.6*(t+1.25)))*stats.norm.pdf(t),-12,12,epsabs=1e-13)[0]
    actual=integrate.quad(lambda t:(m-.1)*expit(-5*(t+6))*stats.norm.pdf(t),-15,15,epsabs=1e-16,points=[-6,-5,-4])[0]
    # Independent normal MGF: E exp(-5 Z)=exp(25/2).
    union=3000*.8*math.exp(-30+25/2)
    return dict(m=m,adaptive_actual_disagreement=actual,analytic_per_person=.8*math.exp(-17.5),
                power_upper=.05+union,condition="size<=.05 at the explicitly feasible null; not uniform size established by a single cell")

def diagnose(kind):
    q=distribution(kind,121);q2=distribution(kind,241);start=time.perf_counter();best=None;attempts=[]
    bounds=[E.BOUNDS[v] for _ in range(12) for v in "abcd"]
    initial=truth_eta();second=initial.copy();second[::4]*=.7
    for z in (initial,second):
        def obj(e):
            if time.perf_counter()-start>120:raise TimeoutError("diagnostic120s budget")
            p,s=E.terms(e,"4PL")
            return -float(q@np.log(p)),-q@s
        try:
            r=optimize.minimize(obj,z,jac=True,method="L-BFGS-B",bounds=bounds,
                options=dict(maxiter=700,maxls=30,ftol=1e-13,gtol=1e-8))
            if best is None or r.fun<best.fun:best=r
            attempts.append(dict(success=bool(r.success),iterations=int(r.nit),message=str(r.message),objective=float(r.fun)))
        except TimeoutError as ex:attempts.append(dict(error=str(ex)))
    if best is None:return dict(kind=kind,attempts=attempts,INCONCLUSIVE=True)
    px,score=E.terms(best.x,"4PL",241);grad=-(q2@score);eta=best.x;b=np.array(bounds)
    pg=grad.copy();pg[(eta<=b[:,0]+1e-7)&(grad>0)]=0;pg[(eta>=b[:,1]-1e-7)&(grad<0)]=0
    KL=float(q2@np.log(q2/px))
    return dict(kind=kind,KL_feasible_null_upper=KL,N_KL=3000*KL,TV_single=float(abs(q2-px).sum()/2),
        theta_GH121_GH241_L1=float(abs(q2-q).sum()),max_projected_gradient=float(abs(pg).max()),
        item1=eta[:4].tolist(),eta=eta.tolist(),attempts=attempts,seconds=time.perf_counter()-start,
        disclaimer="Local optimizer; KL is an upper bound on the closest admissible null KL, not a certified global distance lower bound.",
        weighted_IRF_RMS=float(np.sqrt(E.K.quad(241)[1]@((first_probability(E.K.quad(241)[0],kind,0)-(.1+.8*expit(.8*(E.K.quad(241)[0]+1.25))))**2))) if kind!="local_dependence" else None)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);z=ap.parse_args()
    report=dict(old_bound=independent_old_bound(),alternatives={k:diagnose(k) for k in ("nonmonotone","local_dependence","two_stage")})
    Path(z.out).parent.mkdir(parents=True,exist_ok=True);Path(z.out).write_text(json.dumps(report,indent=1))
    print("DIAGNOSTIC_JSON "+json.dumps(report),flush=True)
if __name__=="__main__":main()
