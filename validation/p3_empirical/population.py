"""Deterministic alternative absorption; no simulated data and no production estimator changes."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy import optimize
from scipy.special import expit,logsumexp,log_expit
from validation.binary_certification.common import truth,PRIORS
from validation.p3_empirical import kernel as E
from validation.p3_empirical.efficient import efficient_basis


def alternative(model,points=121):
    p=truth(model,12);t,w=E.K.quad(points)
    probs=p["c"]+(p["d"]-p["c"])*expit(p["a"]*(t[:,None]-p["b"]))
    mean=float(w@(p["c"][0]+(p["d"][0]-p["c"][0])*expit(2*p["a"][0]*(t-p["b"][0]))))
    logits=p["a"]*(t[:,None]-p["b"])
    with np.errstate(divide="ignore"):
        lp=np.logaddexp(np.log(p["c"]),np.log(p["d"]-p["c"])+log_expit(logits))
        lq=np.logaddexp(np.log1p(-p["d"]),np.log(p["d"]-p["c"])+log_expit(-logits))
    lp[:,0]=np.log(mean);lq[:,0]=np.log1p(-mean)
    logp=logsumexp(E.X@lp.T+(1-E.X)@lq.T+np.log(w),axis=1)
    return np.exp(logp),mean


def run(model,initial):
    start=time.perf_counter();q,m=alternative(model,121);q2,_=alternative(model,241);k=4 if model=="4PL" else 3
    bounds=[E.BOUNDS[v] for _ in range(12) for v in "abcd"[:k]]
    # Beta priors have open endpoint support; numerical objective guards are recorded,
    # not reinterpreted as alternative physical estimation bounds.
    numerical=[(max(lo,1e-10) if i%k==2 else lo,min(hi,1-1e-10) if i%k==3 else hi) for i,(lo,hi) in enumerate(bounds)]
    def prior(eta):
        a,b,c,d=E.K.unpack(eta,model)
        return sum(float(PRIORS[v].logpdf(z).sum()) for v,z in zip("abcd",(a,b,c,d)) if v!="d" or model=="4PL")
    def objective(eta):
        px,score=E.terms(eta,model)
        gp,_=E.K.log_prior_grad_hess(eta,model)
        return -float(q@np.log(px)+prior(eta)/3000),-(q@score+gp/3000)
    fit=optimize.minimize(objective,initial,jac=True,method="L-BFGS-B",bounds=numerical,
        options={"maxiter":1500,"maxls":40,"gtol":1e-9,"ftol":1e-14})
    eta=fit.x;px,score=E.terms(eta,model);val,grad=objective(eta)
    pg=grad.copy();bounds=np.array(bounds)
    pg[(eta<=bounds[:,0]+1e-7)&(grad>0)]=0;pg[(eta>=bounds[:,1]-1e-7)&(grad<0)]=0
    KL=float(q@np.log(q/px));cond,dc=E.conditionals(eta,model)
    H=np.stack([(E.S==s)*(E.X[:,0]-cond[E.S,0]) for s in range(1,12)],1)
    T,Sig,orth=efficient_basis(px,score,H);ev,V=np.linalg.eigh(Sig);pos=ev>1e-10*ev.max()
    u=V.T@(q@T);ncp=float(3000*np.sum(u[pos]**2/ev[pos]))
    return {"model":model,"status":str(fit.message),"success":bool(fit.success),"iterations":int(fit.nit),
        "seconds":time.perf_counter()-start,"max_projected_gradient":float(abs(pg).max()),"eta":eta.tolist(),
        "item1":dict(zip("abcd"[:k],eta[:k].tolist())),"independent_item_marginal":m,
        "alternative_GH121_GH241_L1":float(abs(q-q2).sum()),"KL_alt_vs_feasible_null":KL,"N_KL":3000*KL,
        "Pinsker_power_upper_for_any_level_.05_test":min(1.,.05+np.sqrt(3000*KL/2)),
        "efficient_score_rank":int(pos.sum()),"efficient_score_noncentrality":ncp,
        "raw_residual_mean_norm":float(np.linalg.norm(q@H)),"efficient_orthogonality":orth,
        "warning":"A feasible-null KL bound is valid without claiming global optimality. Optimization failure does not prove structural nonidentifiability. Historical item1 is independent of all other items, not doubled slope on shared theta."}


def main():
    a=argparse.ArgumentParser();a.add_argument("--input",required=True);a.add_argument("--out",required=True);z=a.parse_args()
    output={}
    for model in ("3PL","4PL"):
        files=sorted(Path(z.input).glob(f"{model}_202611261002_*.npz"))
        initial=np.load(files[0])["eta"]
        output[model]=run(model,initial)
    Path(z.out).parent.mkdir(parents=True,exist_ok=True);Path(z.out).write_text(json.dumps(output,indent=1))
    print("POPULATION_DIAG_JSON "+json.dumps(output),flush=True)


if __name__=="__main__":main()
