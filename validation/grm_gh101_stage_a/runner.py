"""Stage A opt-in runner; production and old evidence read-only."""
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
from unittest.mock import patch
import numpy as np
import pandas as pd
from core import grm_dif as G
from validation.gh_reoptimization import experiment as E
from validation.gh81_diagnostic import diagnostic as D
from validation.grm_gh101_uniform_pilot.pilot import layouts, cold_start, PilotStop
from validation.ordinal_dif_paired.control import TraceRNG
from validation.dif_validation_study import gen_ordinal
from validation.grm_gh101_stage_a.statistics import decisions, summarize
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE="cfbb1992e53f992c321bd39fe617fb6b772bccd9"
PROTOCOL_COMMIT="f4f3c237fc5b04407d8644bd0fbd7215f244d5a2"

def load_matrix(old):
    path=HERE/"results"/old["matrix_file"]
    if D.sha(path.read_bytes())!=old["matrix_sha256"]:raise ValueError("matrix file SHA256")
    with np.load(path) as f:x,g=f["X"].copy(),f["group"].copy()
    if D.sha(x.tobytes()+g.tobytes())!=old["data_sha256"]:raise ValueError("matrix content SHA256")
    return x,g

def dense_information(p, design, parts, grid, n):
    matrix=np.empty((design.size,design.size))
    for j in range(design.size):
        h=1e-5*max(1.,abs(p[j]));step=np.zeros(design.size);step[j]=h
        matrix[:,j]=n*(G.objective(p+step,design,parts,grid)[1]-G.objective(p-step,design,parts,grid)[1])/(2*h)
    asymmetry=float(np.max(np.abs(matrix-matrix.T)));matrix=(matrix+matrix.T)/2
    eigen=np.linalg.eigvalsh(matrix)
    return dict(valid=bool(np.isfinite(eigen).all() and eigen[0]>1e-7*max(eigen[-1],1.)),
        matrix=matrix,eigenvalues=eigen,min_eigenvalue=eigen[0],condition=eigen[-1]/eigen[0] if eigen[0]>0 else None,
        max_asymmetry_before_symmetrizing=asymmetry)


def run_case(seed,impact,old,checkpoint):
    x,g=load_matrix(old);columns=[f'P{i:02d}' for i in range(1,11)]
    inputs=G.validate_inputs(pd.DataFrame(x,columns=columns),g,0,1,columns[:8],categories=4,min_cell=5)
    fitter=G.Fitter(inputs,points=101);fitter.audit_grid=D.reference_grid(161)
    designs,slots=layouts();result=dict(seed=seed,impact=impact,data_sha256=old['data_sha256'],slots=slots,solutions={},lr=[],valid=False,
        fits_attempted=0,fits_completed=0,optimizer_calls=[],initialization='native default, start_fit=None for every unique hypothesis')
    checkpoint(result)
    for key,m in designs.items():
        d=m['design'];expected,clipped=cold_start(d,inputs)
        result['active_model']=dict(solution=key,slot=m['first_slot']);result['fits_attempted']+=1;checkpoint(result)
        original=G.minimize
        def tracked(fun,x0,*args,**kwargs):
            call=dict(solution=key,x0=np.asarray(x0).copy(),expected_cold_sha256=D.sha(expected.tobytes()),cold_verified=bool(np.array_equal(x0,expected)),
                      native_initialization_clipped=clipped,method=kwargs.get('method'),options=kwargs.get('options'),jac=kwargs.get('jac'))
            result['optimizer_calls'].append(call);checkpoint(result)
            if not call['cold_verified']:raise PilotStop('Actual optimizer start is not native cold default')
            fitted=original(fun,x0,*args,**kwargs)
            call.update(scipy_success=bool(fitted.success),scipy_status=int(fitted.status),scipy_message=str(fitted.message),nfev=int(fitted.nfev),njev=int(fitted.njev))
            return fitted
        t=time.monotonic()
        with patch.object(G,'minimize',tracked):
            f=fitter.fit(d.roles,start_fit=None,information=True,max_iter=2000)
        result['fits_completed']+=1
        a,b,mu,sd=d.unpack(f['params'])
        s=dict(first_slot=m['first_slot'],roles=d.roles,params=f['params'].copy(),a=a,b=b,focal_mean=mu,focal_sd=sd,
               converged=f['converged'],at_bound=f['at_bound'],native_score=f['score'],audit_score=f['audit_score'],quadrature_ok=f['quadrature_ok'],quadrature_error=f['quadrature_error'],
               native_information_min_eigenvalue=f['information_min_eigenvalue'],native_information_condition=f['information_condition'],covariance_available=f['cov'] is not None,
               anchor_equality=E.anchor_feasible(f),reason=f['reason'],iterations=f['iterations'],fit_seconds=time.monotonic()-t,passed=False)
        result['solutions'][key]=s;checkpoint(result)
        native_ok=bool(f['converged'] and not f['at_bound'] and f['quadrature_ok'] and f['cov'] is not None and s['anchor_equality'])
        if not native_ok:result['failure_reason']='Native convergence/bounds/information/quadrature/anchor gate';return result
        evaluations={}
        for q,grid in ((101,fitter.grid),(161,fitter.audit_grid)):
            value,score=G.objective(f['params'],d,fitter.parts,grid);j=int(np.argmax(np.abs(score)))
            evaluations[str(q)]=dict(loglik=-len(x)*value,gradient=score,max_abs_normalized_gradient=float(abs(score[j])),max_parameter=D.labels(d)[j],
                independent_loglik=-len(x)*D.reference_nll(f['params'],d,x,g,q))
        dense=dense_information(f['params'],d,fitter.parts,fitter.audit_grid,len(x))
        derivative_checks={};derivatives_ok=True
        for h in (1e-4,1e-5):
            fd,_=D.finite_difference(f['params'],d,x,g,161,h);analytic=evaluations['161']['gradient']
            derivative_checks[str(h)]=dict(gradient=fd,max_abs_error=float(np.max(np.abs(fd-analytic))))
            derivatives_ok &= bool(np.allclose(fd,analytic,atol=1e-8,rtol=1e-5))
        s.update(evaluations=evaluations,dense_information=dense,derivative_checks=derivative_checks,
                 derivatives_ok=derivatives_ok,passed=bool(dense['valid'] and derivatives_ok))
        checkpoint(result)
        if not s['passed']:result['failure_reason']='Dense information or independent derivative gate';return result
        result.pop('active_model',None)
    if len(result['optimizer_calls'])!=7 or fitter.optimizations!=7:result['failure_reason']='Unexpected cache/optimizer call count';return result
    table=E.lr_table(slots,result['solutions'],101,161)
    result['lr']=table;result['valid']=all(r['valid'] for r in table)
    if not result['valid']:result['failure_reason']='Native nested-LR gate'
    result['native_optimizer_calls']=fitter.optimizations
    return result


def ordinal_records():
    rows={}
    for impact,folder in (("minus","ordinal_dif_total_rest"),("zero","ordinal_dif_impact_zero")):
        path=ROOT/"validation"/folder/"results/records.jsonl.gz"
        for line in gzip.decompress(path.read_bytes()).splitlines():
            r=json.loads(line)
            if 2026110100<=r["seed"]<2026110200:rows[r["seed"],impact]=r
    if len(rows)!=200:raise ValueError("Missing archived ordinal seed family")
    return rows


def prepare_data(ordinal,output):
    pairs=[];spec={}
    for seed in range(2026110100,2026110200):
        traces={};arrays={}
        for impact,mu in (("minus",-.5),("zero",0.)):
            trace=TraceRNG(seed)
            with patch("numpy.random.default_rng",return_value=trace):frame,group=gen_ordinal(500,500,mu,False,seed)
            x,g=frame.to_numpy(dtype=np.uint8),group.astype(np.uint8)
            digest=D.sha(x.tobytes()+g.tobytes())
            if digest!=ordinal[seed,impact]["data_sha256"]:raise ValueError("archived ordinal response hash mismatch")
            traces[impact]=trace.calls;arrays[impact]=(x,g)
            relative=f"matrices/{seed}_{impact}.npz";path=output/relative;path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,X=x,group=g)
            spec[seed,impact]=dict(data_sha256=digest,matrix_file=relative,matrix_sha256=D.sha(path.read_bytes()))
        expected=[("standard_normal",[500])]*2+[("random",[1000])]*10
        if traces["minus"]!=traces["zero"] or [(v["method"],v["shape"]) for v in traces["minus"]]!=expected:raise ValueError("RNG pairing mismatch")
        minus,gm=arrays["minus"];zero,gz=arrays["zero"]
        if not np.array_equal(gm,gz) or not np.array_equal(minus[:500],zero[:500]) or not (zero[500:]>=minus[500:]).all():raise ValueError("common random pairing mismatch")
        pairs.append(dict(seed=seed,draws=traces["minus"],groups_identical=True,reference_identical=True,focal_monotone=True))
    return spec,pairs


def source_manifest():
    paths=list((ROOT/"core").glob("*.py"))
    paths += [ROOT/v for v in ("validation/gh81_diagnostic/diagnostic.py","validation/gh_reoptimization/experiment.py","validation/grm_gh101_uniform_pilot/pilot.py","validation/ordinal_dif_paired/experiment.py","validation/ordinal_dif_paired/control.py","validation/dif_validation_study.py","validation/coverage_precision.py","validation/ordinal_dif_impact_zero/experiment.py","validation/ordinal_dif_total_rest/experiment.py")]
    paths += list(HERE.glob("*.py"))+[HERE/"protocol.json"]
    return {str(p.relative_to(ROOT)):D.sha(p.read_bytes()) for p in sorted(paths)}


def run():
    output=HERE/"results"
    if (output/"metadata.json").exists():raise RuntimeError("Evidence exists; no repeat/retry")
    output.mkdir(exist_ok=True)
    start=time.monotonic();protocol=json.loads((HERE/"protocol.json").read_text());rows=[];stop="fixed100 pairs complete"
    versions={p:importlib.metadata.version(p) for p in ("numpy","scipy","pandas")}
    if any(versions[p]!=protocol["environment"][p] for p in versions):raise ValueError("Runtime differs from preregistration")
    if protocol["seeds"]!=list(range(2026110100,2026110200)):raise ValueError("Seed plan changed")
    archived=ordinal_records();spec,pairs=prepare_data(archived,output);sources=source_manifest()
    E.save(output/"metadata.json",dict(protocol_sha256=D.sha((HERE/"protocol.json").read_bytes()),protocol_commit=PROTOCOL_COMMIT,base_commit=BASE,execution_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),versions=versions,python=os.sys.version,threads={k:os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")},sources=sources,pairing=pairs,pilot_reused=0,ordinal_sources={folder:D.sha((ROOT/"validation"/folder/"results/records.jsonl.gz").read_bytes()) for folder in ("ordinal_dif_total_rest","ordinal_dif_impact_zero")}))
    current=None;t=start
    def checkpoint(case):
        nonlocal current
        current=case;E.save(output/"active_case.json",case)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024>1024:raise PilotStop("RSS limit")
    def alarm(*a):raise TimeoutError("per-case120s / total7200s")
    old=signal.signal(signal.SIGALRM,alarm)
    try:
        for plan in protocol["analysis_order"]:
            seed,impact=plan["seed"],plan["impact"];remaining=7200-(time.monotonic()-start)
            if remaining<=0:stop="Total wall limit";break
            current=dict(seed=seed,impact=impact,valid=False,solutions={},fits_attempted=0,fits_completed=0,**spec[seed,impact])
            t=time.monotonic();signal.setitimer(signal.ITIMER_REAL,min(120,remaining));fatal=False
            try:case=run_case(seed,impact,spec[seed,impact],checkpoint)
            except TimeoutError as exc:
                case=current;case.update(valid=False,failure_reason=str(exc),timeout=True)
            except Exception as exc:
                case=current;case.update(valid=False,failure_reason=f"{type(exc).__name__}: {exc}");fatal=True
            finally:signal.setitimer(signal.ITIMER_REAL,0)
            case.update(**spec[seed,impact],wall_seconds=time.monotonic()-t,peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            case["decisions"]=decisions(case);rows.append(case)
            with (output/"records.jsonl").open("a") as stream:
                stream.write(json.dumps(E.clean(case),allow_nan=False)+"\n");stream.flush();os.fsync(stream.fileno())
            current=None;(output/"active_case.json").unlink(missing_ok=True)
            E.save(output/"progress.json",dict(attempted=len(rows),valid=sum(r["valid"] for r in rows),seconds=time.monotonic()-start))
            print(f'{len(rows)}/200 {seed} {impact} valid={case["valid"]} {case["wall_seconds"]:.2f}s RSS={case["peak_process_rss_mib"]:.1f}MiB',flush=True)
            if fatal or case["peak_process_rss_mib"]>1024:stop=case.get("failure_reason","RSS limit");break
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    if source_manifest()!=sources:raise ValueError("Source changed during execution")
    raw=(output/"records.jsonl").read_bytes() if (output/"records.jsonl").exists() else b""
    packed=gzip.compress(raw,mtime=0);assert gzip.decompress(packed)==raw
    (output/"records.jsonl.gz").write_bytes(packed);(output/"records.jsonl").unlink(missing_ok=True)
    E.save(output/"ordinal.json",[dict(seed=s,impact=k,data_sha256=v["data_sha256"],methods=v["methods"]) for (s,k),v in archived.items()])
    summary=summarize(rows,archived);summary.update(stop=stop,execution_seconds=time.monotonic()-start,native_fit_seconds=sum(s.get("fit_seconds",0) for r in rows for s in r["solutions"].values()),peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,records_sha256=D.sha(raw),records_gzip_sha256=D.sha(packed),old_mc_pooled=False,pilot_reused=0)
    E.save(output/"summary.json",summary)
    from validation.grm_gh101_stage_a.audit import audit
    E.save(output/"audit.json",audit(output))
    summary["total_seconds_with_audit"]=time.monotonic()-start
    summary["peak_process_rss_mib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if summary["total_seconds_with_audit"]>7200 or summary["peak_process_rss_mib"]>1024:
        summary["stop"]="Stage resource limit exceeded including audit"
    E.save(output/"summary.json",summary)
    E.save(output/"SHA256.json",{str(p.relative_to(output)):D.sha(p.read_bytes()) for p in sorted(output.rglob("*")) if p.is_file() and p.name!="SHA256.json"})
    print(json.dumps({k:summary[k] for k in ("attempted","valid","invalid","not_run","stop","total_seconds_with_audit","peak_process_rss_mib")}),flush=True)

if __name__=="__main__":run()
