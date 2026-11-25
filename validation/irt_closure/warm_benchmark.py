"""Paired development-only native refit speed/equivalence benchmark; no confirmation reuse."""
import json,time
from pathlib import Path
import numpy as np
from core.irt_python_mirt_adapter import fit_python_mirt
from validation.binary_certification.common import Objective
from validation.irt_closure.patterns import PATTERNS,moments,probability
from validation.irt_closure.engine import native_statistics,coefficients,write_json,COLS

def benchmark(model,work):
    work=Path(work);qs="abc" if model=="3PL" else "abcd";records=[]
    for method in ("warm","warm-squarem"):
        cases=[]
        for r in (0,4):
            base=dict(np.load(work/f"base_{r:05d}.npz",allow_pickle=False))
            cached=dict(np.load(work/f"boot_{r:05d}.npz",allow_pickle=False))
            p={q:base[q] for q in "abcd"}
            mass,_,_=moments(p,241);fine,_,_=moments(p,481)
            use_counts=float(abs(mass-fine).sum())<=1e-8;mass=fine/fine.sum()
            for b in range(5):
                if not cached["converged"][b]:
                    cases.append({"r":r,"b":b,"error":"cold reference nonconvergence","P1":False,"P3":False});break
                rng=np.random.default_rng([202611251001,int(model[0]),r,b+1])
                if use_counts:
                    counts=rng.multinomial(3000,mass);x=np.repeat(PATTERNS,counts,axis=0)
                else:
                    theta=rng.standard_normal(3000);pr=probability(theta,p)
                    x=(rng.random(pr.shape)<pr).astype(np.uint8)
                cold=np.asarray(cached["estimates"][b]).reshape(12,len(qs))
                coldp={q:cold[:,i] for i,q in enumerate(qs)}
                start={COLS[q]:p[q] for q in qs}
                t0=time.monotonic()
                try:
                    outcome=fit_python_mirt(x,model=model,estimation="MAP",n_quadpts=61,
                        max_iter=500,tol=1e-4,start_values=start,
                        accelerate="squarem" if method=="warm-squarem" else "none")
                    f=outcome.fit;seconds=time.monotonic()-t0;warm=coefficients(f,model)
                    stat,_,_,_=native_statistics(f,x)
                    obj=Objective(x,model,map_priors=True,points=61)
                    gap=abs(obj.value(coldp)-obj.value({q:warm[q] for q in qs}))
                    diff={q:float(abs(coldp[q]-warm[q]).max()) for q in qs}
                    sx_error=float(abs(stat-cached["sx"][b]).max())
                    p1=bool(f.converged) and gap<=.001 and all(v<=(.002 if q in "ab" else .001) for q,v in diff.items())
                    p3=bool(f.converged) and gap<=.001 and np.isfinite(sx_error) and sx_error<=.0022
                    cases.append({"r":r,"b":b,"seconds":seconds,"converged":bool(f.converged),
                        "parameter_max":diff,"objective_error":float(gap),"SX2_error":sx_error,"P1":p1,"P3":p3})
                except Exception as e:cases.append({"r":r,"b":b,"error":repr(e),"P1":False,"P3":False})
                if not cases[-1]["P1"] and not cases[-1]["P3"]:break
            if cases and not cases[-1]["P1"] and not cases[-1]["P3"]:break
        records.append({"method":method,"cases":cases,"checked":len(cases),
            "P1_eligible":len(cases)==10 and all(x["P1"] for x in cases),
            "P3_eligible":len(cases)==10 and all(x["P3"] for x in cases),
            "mean_seconds":float(np.mean([x["seconds"] for x in cases if "seconds" in x])) if any("seconds" in x for x in cases) else None})
    output={"model":model,"development_only":True,"certification":"NOT CERTIFIED","candidates":records}
    write_json(work/"warm_benchmark.json",output)
    print("BENCHMARK_JSON "+json.dumps(output))
    return output

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("--model",required=True);ap.add_argument("--work",required=True)
    a=ap.parse_args();benchmark(a.model,a.work)
