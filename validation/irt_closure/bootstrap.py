"""Full-refit bootstrap binding inversion; native MAP point is unchanged."""
import json,time
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
from mirt.estimation.base import _ITEM_PARAMETER_BOUNDS
from validation.parameter_uncertainty_g.runner import marginal_scores
from validation.binary_certification.reference_numpy import PRIOR
from validation.irt_closure.patterns import PATTERNS,moments,probability
from validation.irt_closure.engine import fit_once,coefficients,native_statistics,deadline,write_json,ROOTS

NAMES={"a":"discrimination","b":"difficulty","c":"guessing","d":"upper"}


def pack(p,model):
    qs="abc" if model=="3PL" else "abcd"
    v=np.column_stack([p[q] for q in qs]).ravel()
    bounds=np.array([_ITEM_PARAMETER_BOUNDS[NAMES[q]] for j in range(12) for q in qs])
    return v,bounds,qs


def eta(v,bounds):
    lo,hi=bounds.T
    if np.any(np.minimum(v-lo,hi-v)<=1e-6):raise ValueError("Native active-bound refusal")
    u=(v-lo)/(hi-lo)
    return logit(u),(hi-lo)*u*(1-u)


def physical(e,bounds):
    lo,hi=bounds.T
    v=lo+(hi-lo)*expit(e)
    if np.any((v<=lo)|(v>=hi)):raise ValueError("Numerical bound saturation")
    return v


def binding_inversion(e,star,scores,V,I,prior):
    if len(star)<2:raise ValueError("Insufficient bootstrap refits")
    linear=e+(scores+prior)@V.T
    residual=star-linear
    centered=residual-residual.mean(0)
    score_centered=scores-scores.mean(0)
    J=V@I+centered.T@score_centered/(len(star)-1)
    if not np.isfinite(J).all() or np.linalg.cond(J)>1e8:raise ValueError("Binding Jacobian condition refusal")
    draws=e+np.linalg.solve(J,(e-star).T).T
    return draws,J


def run_bootstrap(model,r,stage,data,work,B):
    path=Path(work)/f"boot_{r:05d}.npz";meta=path.with_suffix(".json")
    p={q:data[q] for q in "abcd"};v,bounds,qs=pack(p,model)
    e,D=eta(v,bounds)
    V=data["V"];ev=np.linalg.eigvalsh((V+V.T)/2)
    if not np.isfinite(V).all() or ev[0]<=0 or ev[-1]/ev[0]>1e8:raise ValueError("Native covariance refusal")
    # Full physical-to-box-logit covariance for a variance-reduction control only;
    # derivative identity remains valid for any fixed nonsingular control matrix.
    Veta=V/D[:,None]/D[None,:]
    mass,_,_=moments(p,241);fine,_,_=moments(p,481)
    quadrature=float(abs(mass-fine).sum())
    mass=fine/fine.sum()
    raw=marginal_scores(PATTERNS,p,points=241)
    U=np.column_stack([raw[q][:,j] for j in range(12) for q in qs])*D
    mean_score=3000*(mass@U)
    I=3000*(U.T@(mass[:,None]*U)-np.outer(mass@U,mass@U))
    prior=np.array([PRIOR[q][1](p[q][j]) for j in range(12) for q in qs])*D
    if path.exists():
        cache=dict(np.load(path,allow_pickle=False));done=len(cache["converged"])
        estimates=list(cache["estimates"]);stats=list(cache["sx"]);converged=list(cache["converged"])
        score_totals=list(cache["scores"]);coord_valid=list(cache["coordinate_valid"]);errors=list(cache["errors"]) if "errors" in cache else ["unknown"]*done
    else:done=0;estimates=[];stats=[];converged=[];score_totals=[];coord_valid=[];errors=[]
    start=time.monotonic()
    for b in range(done,B):
        deadline()
        rng=np.random.default_rng([ROOTS[stage],int(model[0]),r,b+1])
        if quadrature<=1e-8:
            counts=rng.multinomial(3000,mass);x=np.repeat(PATTERNS,counts,axis=0)
        else:
            th=rng.standard_normal(3000);pr=probability(th,p)
            x=(rng.random(pr.shape)<pr).astype(np.uint8)
            codes=(x*(1<<np.arange(12))).sum(1);counts=np.bincount(codes,minlength=4096)
        estimate=np.full(len(e),np.nan);sx=np.full(12,np.nan);ok=False;coordinate=False;error=""
        try:
            out=fit_once(x,model);f=out.fit
            ok=bool(f.converged)
            if ok:
                bootp=coefficients(f,model);phys,_,_=pack(bootp,model)
                estimate=phys
                sx,_,_,_=native_statistics(f,x)
                eta(phys,bounds);coordinate=True
        except Exception as exc:error=repr(exc)  # retained; no replacement/retry
        estimates.append(estimate);stats.append(sx);converged.append(ok)
        score_totals.append(counts@U-mean_score);coord_valid.append(coordinate);errors.append(error)
        if True:  # checkpoint every completed refit
            np.savez_compressed(path,estimates=estimates,sx=stats,converged=converged,
                scores=score_totals,coordinate_valid=coord_valid,errors=np.array(errors,dtype=str))
    estimates=np.asarray(estimates);scores=np.asarray(score_totals);ok=np.asarray(converged)&np.asarray(coord_valid)
    result={"B":B,"completed":len(ok),"converged":int(np.sum(converged)),"eligible":int(ok.sum()),
        "inner_refusal":float(1-ok.mean()),"pattern_mass_241_481_L1":quadrature,
        "seconds_new":time.monotonic()-start,"point_MAP":v.tolist(),"bounds":bounds.tolist(),
        "p1_valid":False,"p3_valid":False,"error_counts":{e:errors.count(e) for e in set(errors) if e}}
    # P3 uses its preregistered first99 draws and treats failed comparisons conservatively.
    stat=np.asarray(stats)[:99];valid=np.isfinite(stat)
    if len(stat)==99:
        exceed=((stat>=data["sx"])|~valid).sum(0)
        result["bootstrap_pvalue"]=((1+exceed)/100).tolist()
        result["p3_non_evaluable_rate"]=(1-valid.mean(0)).tolist()
        result["p3_valid"]=bool(np.all(1-valid.mean(0)<=.01))
    if len(ok)==B and 1-ok.mean()<=.01:
        stars=np.array([eta(x,bounds)[0] for x in estimates[ok]])
        draws,J=binding_inversion(e,stars,scores[ok],Veta,I,prior)
        low_eta,high_eta=np.quantile(draws,[.025,.975],axis=0)
        low=physical(low_eta,bounds);high=physical(high_eta,bounds)
        from validation.binary_certification.common import truth
        target,_bounds,_=pack(truth(model,12),model)
        result.update(p1_valid=True,qs=qs,estimate=v.tolist(),
            sampling_se=np.std(estimates[ok],axis=0,ddof=1).tolist(),
            ci_lower=low.tolist(),ci_upper=high.tolist(),covered=((low<=target)&(target<=high)).astype(int).tolist(),
            error=(v-target).tolist(),binding_condition=float(np.linalg.cond(J)),
            binding_center_eta=(e+np.linalg.solve(J,e-stars.mean(0))).tolist())
    write_json(meta,result)
    return result
