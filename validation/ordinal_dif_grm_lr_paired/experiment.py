"""Opt-in, budgeted GRM-LR feasibility; no ordinal refits or production changes."""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.stats import chi2
from core import grm_dif as G
from core.dif_classical import adjust_pvalues
from validation.dif_validation_study import gen_ordinal
from validation.ordinal_dif_paired import experiment as P
from validation.ordinal_dif_impact_zero import experiment as Z

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ANCHORS=[f'P{i:02d}' for i in range(1,9)]


def protect():
    manifest=json.loads((HERE/'prior_manifest.json').read_text())
    for name,value in manifest.items():
        if Z.digest((ROOT/name).read_bytes())!=value:raise ValueError(f'Prior canonical evidence changed: {name}')
    return len(manifest)


def gates(result):
    audits={};all_ok=True
    for item,fits in result.get('fits',{}).items():
        audits[item]={}
        for label,f in fits.items():
            a,b=np.asarray(f['a']),np.asarray(f['b']);gap=np.diff(b,axis=2)
            anchors=bool(np.array_equal(a[0,:8],a[1,:8]) and np.array_equal(b[0,:8],b[1,:8]))
            feasible=bool(np.all((a>=.1)&(a<=4)) and np.all((b[:,:,0]>=-6)&(b[:,:,0]<=4)) and np.all((gap>=.05)&(gap<=6)) and -3<=f['focal_mean']<=3 and .3<=f['focal_sd']<=3)
            mineig=f.get('information_min_eigenvalue');cond=f.get('information_condition')
            information=bool(mineig is not None and cond is not None and np.isfinite([mineig,cond]).all() and mineig>1e-7*max(mineig*cond,1.) and f.get('information_available',False))
            regular=bool(f['converged'] and not f['at_bound'] and f['score']<2e-6 and f['quadrature_ok'] and f['quadrature_error']<=1e-3 and f['audit_score']<=1e-5)
            audits[item][label]=dict(anchor_equality=anchors,physical_feasible=feasible,information=information,native_regular=regular)
            all_ok &= anchors and feasible and information and regular
    layout=set(result.get('fits',{}))=={'P09','P10'} and all(set(f)=={'H0','Hb','Ha','H2'} for f in result.get('fits',{}).values())
    df_ok=True;lr_ok=True
    for row in result.get('table',[]):
        df_ok &= row['sd (genel)']==4
        lr=2*(row['logL (H2)']-row['logL (H0)'])
        # Arithmetic verification only; native negative-LR/inference gates remain authoritative.
        lr_ok &= lr>=-1e-6 and math.isclose(max(0.,lr),row['LR (genel)'],rel_tol=1e-10,abs_tol=1e-9)
        if row['p'] is not None:lr_ok &= math.isclose(float(chi2.sf(max(0.,lr),4)),row['p'],rel_tol=1e-10,abs_tol=1e-14)
    return dict(valid=bool(result.get('valid',False) and all_ok and layout and df_ok and lr_ok),fits=audits,layout=layout,df4=bool(df_ok),lr_arithmetic=bool(lr_ok))


def worker(source,destination):
    P.grm_worker(source,destination)  # Existing PR118 production wrapper; categories INTEGER4, GH61.
    result=json.loads(destination.read_text())
    result['gate_audit']=gates(result)
    result['native_family_valid']=result.get('valid',False)
    result['valid']=result['gate_audit']['valid']
    result['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    P.save(destination,result)


def comparisons(records,cached):
    out={};names=[]
    for impact in ('minus','zero'):
        for method in ('total','rest'):
            rows=[r for r in records if r['impact']==impact and r['result'].get('valid',False) and cached[impact][r['seed']]['methods'][method]['valid']]
            r=Z.contrast([v['result']['holm_event'] for v in rows],[cached[impact][v['seed']]['methods'][method]['holm_event'] for v in rows],sum(v['impact']==impact for v in records))
            r['joint_nine_ci']=Z.mean_stats([int(v['result']['holm_event'])-int(cached[impact][v['seed']]['methods'][method]['holm_event']) for v in rows],2,.05/9)['hoeffding_ci']
            r.pop('simultaneous_four_contrasts_ci',None)
            name=f'GRM-minus-{method}_{impact}';out[name]=r;names.append(name)
    by={(r['seed'],r['impact']):r for r in records}
    seeds=sorted({r['seed'] for r in records})
    rows=[(by[(s,'zero')],by[(s,'minus')]) for s in seeds if all((s,k) in by and by[(s,k)]['result'].get('valid',False) for k in ('zero','minus'))]
    r=Z.contrast([a['result']['holm_event'] for a,b in rows],[b['result']['holm_event'] for a,b in rows],len(seeds))
    r['joint_nine_ci']=Z.mean_stats([int(a['result']['holm_event'])-int(b['result']['holm_event']) for a,b in rows],2,.05/9)['hoeffding_ci'];r.pop('simultaneous_four_contrasts_ci',None)
    out['GRM_zero-minus-minus']=r;names.append('GRM_zero-minus-minus')
    valid_p=[out[n]['mcnemar_p'] for n in names]
    if all(p is not None for p in valid_p):
        for name,p in zip(names,adjust_pvalues(valid_p,'Holm')):out[name]['mcnemar_holm_five_p']=float(p)
    for method in ('total','rest'):
        quads=[]
        for a,b in rows:
            seed=a['seed'];z=cached['zero'][seed]['methods'][method];m=cached['minus'][seed]['methods'][method]
            if z['valid'] and m['valid']:quads.append((int(a['result']['holm_event']),int(z['holm_event']),int(b['result']['holm_event']),int(m['holm_event'])))
        values=[a-b-c+d for a,b,c,d in quads]
        r=Z.mean_stats(values,4);r['joint_nine_ci']=Z.mean_stats(values,4,.05/9)['hoeffding_ci']
        import itertools
        r['quadruple_order']=['GRM0',method+'0','GRMm',method+'m']
        r['quadruple_table']={''.join(map(str,q)):quads.count(q) for q in itertools.product((0,1),repeat=4)}
        n=len(quads)
        def interval(x,y):
            return Z.exact_paired_ci(sum(a==1 and b==0 for a,b in zip(x,y)),sum(a==0 and b==1 for a,b in zip(x,y)),n,.025)
        zi=interval([q[0] for q in quads],[q[1] for q in quads]);mi=interval([q[2] for q in quads],[q[3] for q in quads])
        r['exact_paired_ci']=[max(-2.,zi[0]-mi[1]),min(2.,zi[1]-mi[0])]
        missing=len(seeds)-n;s=sum(values)
        r['unknown_failure_empirical_bounds']=[(s-2*missing)/len(seeds),(s+2*missing)/len(seeds)] if seeds else [-2,2]
        r['mcnemar']='NOT APPLICABLE: nonbinary interaction'
        out['interaction_GRM-minus-'+method]=r
    return out


def summarize(records,cached,elapsed,stop):
    summary=dict(attempted=len(records),valid=sum(r['result'].get('valid',False) for r in records),seconds=elapsed,stop=stop,scientific_status='FEASIBILITY ONLY / NOT CERTIFIED',conditions={})
    for impact in ('minus','zero'):
        rows=[r for r in records if r['impact']==impact];valid=[r for r in rows if r['result'].get('valid',False)]
        n=len(valid);events=sum(r['result']['holm_event'] for r in valid);bh=sum(r['result']['bh_event'] for r in valid)
        t=sum(r['wall_seconds'] for r in rows)
        summary['conditions'][impact]=dict(attempted=len(rows),valid=n,failed=len(rows)-n,native_converged=sum(r['all_native_converged'] for r in rows),
            holm_events=events,fwer=events/n if n else None,fwer_cp975=P.binomial_ci(events,n,.025),joint_nine_cp=P.binomial_ci(events,n,.05/9),
            fwer_mcse=math.sqrt((events/n)*(1-events/n)/(n-1)) if n>1 else None,
            bh_global_null_fdr=bh/n if n else None,raw_type1=float(np.mean([v['result']['raw_rejections'] for v in valid])) if n else None,
            seconds=t,mean_seconds=t/len(rows) if rows else None,projected_2500_seconds=t/len(rows)*2500 if rows else None,
            failed_outcome_empirical_bounds=[events/len(rows),(events+len(rows)-n)/len(rows)] if rows else [0,1],
            raw_type1_cluster_mcse=float(np.std([np.mean(v['result']['raw_rejections']) for v in valid],ddof=1)/np.sqrt(n)) if n>1 else None)
    summary['mean_seconds_per_analysis']=elapsed/len(records) if records else None
    summary['projected_2500_pairs_both_impacts_seconds']=elapsed/len(records)*5000 if records else None
    summary['peak_worker_rss_mib']=max((r['result'].get('peak_rss_mib',0.) for r in records),default=0.)
    summary['feasible_for_staged_future_study']=len(records)==20 and summary['valid']==20 and stop=='planned feasibility complete'
    summary['main_within_current_600s_budget']=bool(summary['feasible_for_staged_future_study'] and summary['projected_2500_pairs_both_impacts_seconds']<=600)
    summary['main_executed']=False
    summary['contrasts']=comparisons(records,cached)
    return summary


def run():
    plan=json.loads((HERE/'plan.json').read_text());protect()
    cached={k:{r['seed']:r for r in Z.load_records(path/'results/records.jsonl.gz')} for k,path in (('minus',Z.OLD),('zero',Z.HERE))}
    meta=json.loads((Z.OLD/'results/metadata.json').read_text())
    versions={p:importlib.metadata.version(p) for p in meta['versions']}
    if versions!=meta['versions']:raise ValueError('Archived generator/engine versions differ')
    for p in ('validation/dif_validation_study.py','core/dif_classical.py'):
        if Z.digest((ROOT/p).read_bytes())!=meta['sources'][p]:raise ValueError('Archived source mismatch')
    out=HERE/'results';out.mkdir(exist_ok=True)
    if (out/'records.json.gz').exists():raise RuntimeError('Do not overwrite/repeat prior fits')
    source_names=['validation/ordinal_dif_grm_lr_paired/experiment.py','validation/ordinal_dif_grm_lr_paired/plan.json','validation/ordinal_dif_grm_lr_paired/prior_manifest.json','core/grm_dif.py','core/rasch.py','core/irt_dif.py','validation/ordinal_dif_paired/experiment.py','validation/ordinal_dif_paired/plan.json','validation/ordinal_dif_impact_zero/experiment.py','validation/ordinal_dif_paired/control.py','validation/dif_validation_study.py']
    P.save(out/'metadata.json',dict(versions=versions,python=sys.version,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources={p:Z.digest((ROOT/p).read_bytes()) for p in source_names},plan=plan,protected_files=protect(),canonical_logs='stdout/stderr.txt stored in archive; no ignored *.log dependency'))
    start=time.monotonic();records=[];stop='planned feasibility complete'
    with tempfile.TemporaryDirectory(prefix='grm-feas-') as directory:
        tmp=Path(directory);logs={}
        for index in range(10):
            seed=2026110100+index
            zero,g,crn=Z.paired_control(seed,cached['minus'][seed]);minus,gm=gen_ordinal(500,500,-.5,False,seed)
            if Z.data_hash(zero,g)!=cached['zero'][seed]['data_sha256'] or Z.data_hash(minus,gm)!=cached['minus'][seed]['data_sha256'] or not np.array_equal(g,gm):raise ValueError('Archived dataset pairing mismatch')
            for impact,data in (('minus',minus),('zero',zero)):
                remaining=600-(time.monotonic()-start)
                if remaining<=0:stop='600s budget';break
                np.savez_compressed(out/f'data_{index:02d}_{impact}.npz',X=data.to_numpy(dtype=np.uint8),group=g.astype(np.uint8))
                frame=data.copy();frame['group']=g;source=tmp/'data.csv';frame.to_csv(source,index=False);destination=tmp/'result.json'
                if destination.exists():destination.unlink()
                t=time.monotonic();process=subprocess.Popen([sys.executable,'-m','validation.ordinal_dif_grm_lr_paired.experiment','--worker',str(source),str(destination)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                timed=False
                try:stdout,stderr=process.communicate(timeout=min(90,remaining))
                except subprocess.TimeoutExpired:
                    timed=True;process.kill();stdout,stderr=process.communicate()
                duration=time.monotonic()-t
                result=json.loads(destination.read_text()) if destination.exists() else dict(valid=False,reason='timeout' if timed else 'worker produced no result')
                if timed or process.returncode:result.update(valid=False,worker_error=True)
                fits=[f for item in result.get('fits',{}).values() for f in item.values()]
                record=dict(seed=seed,index=index,impact=impact,data_sha256=Z.data_hash(data,g),matrix_file=f'data_{index:02d}_{impact}.npz',matrix_sha256=Z.digest((out/f'data_{index:02d}_{impact}.npz').read_bytes()),crn=crn,result=result,wall_seconds=duration,returncode=process.returncode,timeout=timed,all_native_converged=bool(len(fits)==8 and all(f['converged'] for f in fits)))
                records.append(record);logs[f'{index:02d}_{impact}']=dict(stdout=stdout,stderr=stderr)
                P.save(out/'progress.json',dict(records=records,logs=logs))
                print(f'Analysis {len(records)}/20 complete; {duration:.2f}s; valid={result.get("valid",False)}',flush=True)
                if result.get('peak_rss_mib',0)>1024:stop='worker memory safety threshold';break
                if time.monotonic()-start>=600:stop='600s budget';break
            if stop!='planned feasibility complete':break
    elapsed=time.monotonic()-start
    payload=json.dumps(P.clean(dict(records=records,logs=logs)),allow_nan=False).encode();packed=gzip.compress(payload,mtime=0)
    assert gzip.decompress(packed)==payload
    (out/'records.json.gz').write_bytes(packed);(out/'progress.json').unlink(missing_ok=True)
    summary=summarize(records,cached,elapsed,stop);summary.update(records_sha256=Z.digest(payload),records_gzip_sha256=Z.digest(packed),prior_files_verified=protect())
    P.save(out/'summary.json',summary)
    print(json.dumps(P.clean(summary),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',nargs=2,type=Path);args=parser.parse_args()
    if args.worker:worker(*args.worker)
    else:run()
