"""Preregistered F pilot: one calibration/model, independent theta holdout; no certification."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np
from scipy.special import expit
from scipy.stats import binomtest
from core.irt_binary_inference import Z95
from validation.binary_certification.common import truth, simulate

PROTOCOL_COMMIT = "72d1d651aa2433fc416cbd960dc7671da9c31e2f"
BASE = Path(__file__).resolve().parent


def coverage_statistics(theta, estimate, sd):
    theta, estimate, sd = [np.asarray(v, float) for v in (theta, estimate, sd)]
    if (theta.ndim != 1 or not len(theta) or theta.shape != estimate.shape
            or theta.shape != sd.shape or not np.isfinite(np.r_[theta, estimate, sd]).all()
            or np.any(sd <= 0)):
        raise ValueError("Complete finite aligned vectors and positive posterior SD required; no exclusions.")
    covered = np.abs(estimate - theta) <= Z95 * sd
    n = len(theta); events = int(covered.sum()); rate = events / n
    interval = binomtest(events, n).proportion_ci(confidence_level=.975, method="exact")
    return {"n": n, "covered": events, "coverage": rate,
            "mcse_conditional_on_calibration": float(np.sqrt(rate * (1-rate)/n)),
            "cp_97_5": [float(interval.low), float(interval.high)],
            "bias": float(np.mean(estimate-theta)),
            "rmse": float(np.sqrt(np.mean((estimate-theta)**2))),
            "coverage_screen_only": bool(abs(rate-.95) <= .02)}, covered


def digest(array):
    arr = np.ascontiguousarray(array)
    return hashlib.sha256(str((arr.shape, arr.dtype.str)).encode()+arr.tobytes()).hexdigest()


def run(model, directory):
    from core.irt_python_mirt_adapter import fit_python_mirt
    from core.irt_binary_persons import binary_person_abilities
    import pandas as pd
    start=time.monotonic(); output=Path(directory); output.mkdir(parents=True, exist_ok=True)
    seed=2026112531 if model=="3PL" else 2026112541
    report={"model":model,"protocol_commit":PROTOCOL_COMMIT,
            "code_commit":os.getenv("GITHUB_SHA"),"run_id":os.getenv("GITHUB_RUN_ID"),
            "versions":{name:importlib.metadata.version(name) for name in ("mirt","numpy","scipy","pandas")},
            "python":platform.python_version(),"status":"NOT EXECUTED",
            "certification":"NOT CERTIFIED","calibration_replicates":1,
            "scope":"Conditional on one independently generated calibration; not unconditional calibration-sampling coverage.",
            "settings":{"N_calibration":3000,"J":12,"N_holdout":10000,
                "seed_calibration":seed,"seed_holdout":seed+1,"estimation":"MAP",
                "n_quadpts":61,"tol":1e-10,"max_iter":50000}}
    try:
        source=(BASE/"PROTOCOL.json").read_bytes()
        report["protocol_sha256"]=hashlib.sha256(source).hexdigest()
        registry=Path("validation/evidence/binary_certification/CERTIFICATION.json").read_bytes()
        report["registry_git_blob"]=hashlib.sha1(b"blob "+str(len(registry)).encode()+b"\0"+registry).hexdigest()
        if report["registry_git_blob"] != "49b5aa914c8ea99ec319ded58bf09ac6b85a4de2":
            raise ValueError("Prior certification registry changed")
        x=simulate(model,3000,12,seed); p=truth(model,12)
        report["calibration_sha256"]=digest(x)
        np.savez_compressed(output/"calibration.npz",responses=x,**p)
        fit_start=time.monotonic()
        outcome=fit_python_mirt(x,model=model,estimation="MAP",n_quadpts=61,tol=1e-10,max_iter=50000)
        report["fit_seconds"]=time.monotonic()-fit_start
        report["fit_converged"]=bool(outcome.fit.converged)
        for key in ("n_iterations","iterations","n_iter","log_likelihood"):
            value=getattr(outcome.fit,key,None)
            if isinstance(value,(float,int,np.floating,np.integer)):
                report[key]=float(value)
        outcome.fit.coef().to_csv(output/"estimated_items.csv",index=False)
        report["true_items"]={q:np.asarray(v).tolist() for q,v in p.items()}
        if not outcome.fit.converged:
            raise ValueError("Single calibration did not converge; no retry")
        rng=np.random.default_rng(seed+1)
        theta=rng.standard_normal(10000)
        probability=p.get("c",0)+(p.get("d",1)-p.get("c",0))*expit(p["a"]*(theta[:,None]-p["b"]))
        responses=(rng.random(probability.shape)<probability).astype(int)
        report["theta_sha256"]=digest(theta);report["holdout_sha256"]=digest(responses)
        np.savez_compressed(output/"holdout.npz",theta=theta,responses=responses)
        ids=[f"H{i:05d}" for i in range(len(theta))]
        people=binary_person_abilities(outcome,responses,ids)
        if people.Birey.tolist()!=ids:
            raise ValueError("Holdout IDs/order changed")
        estimates=people["Yetenek (θ)"].to_numpy(float);sd=people["Posterior SS (θ)"].to_numpy(float)
        report["statistics"],covered=coverage_statistics(theta,estimates,sd)
        report["correlation"]=float(np.corrcoef(theta,estimates)[0,1])
        report["quadrature_max_difference"]=float(people["Quadrature max fark"].max())
        report["theta_bands"]=[]
        for lo,hi in ((-np.inf,-2),(-2,-1),(-1,0),(0,1),(1,2),(2,np.inf)):
            mask=(theta>=lo)&(theta<hi)
            if mask.any():
                stats,_=coverage_statistics(theta[mask],estimates[mask],sd[mask])
                report["theta_bands"].append({"lower":str(lo),"upper":str(hi),**stats})
        people["True theta"]=theta
        people["Diagnostic normal lower"]=estimates-Z95*sd
        people["Diagnostic normal upper"]=estimates+Z95*sd
        people["Covered"]=covered
        people.to_csv(output/"holdout_scores.csv",index=False)
        report["status"]="MEASURED / NOT CERTIFIED"
        report["software_checks"]="PASS"
        report["limitations"]=[
            "One calibration only; 10000 people are not 10000 independent calibration repetitions.",
            "EAP plus/minus 1.96 posterior SD is a diagnostic normal approximation, not an equal-tailed posterior interval.",
            "CP/MCSE cover holdout sampling conditional on this calibration only.",
            "Theta-band coverage can differ; no uniform-in-theta claim.",
            "Only J12; no J30 or rectangle certification; no parameter-SE or S-X2 p certification."]
        ok=True
    except Exception as exc:
        report["status"]="FAIL";report["error"]=repr(exc)
        (output/"stderr.txt").write_text(traceback.format_exc())
        ok=False
    finally:
        report["wall_seconds"]=time.monotonic()-start
        (output/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
        print(json.dumps(report,ensure_ascii=False,allow_nan=False))
    return ok


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",choices=["3PL","4PL"],required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    raise SystemExit(0 if run(args.model,args.output) else 1)
