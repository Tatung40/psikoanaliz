"""Controlled user-stop postprocessor. Reads evidence only; never fits a model."""
import copy
import gzip
import json
from pathlib import Path
import resource
import shutil
import time
import numpy as np
from validation.grm_gh101_stage_a import runner as R
from validation.grm_gh101_stage_a import statistics as S
from validation.grm_gh101_stage_a import audit as A

HERE=Path(__file__).resolve().parent
SEEDS=tuple(range(2026110100,2026110130))


def retarget30(base):
    result=copy.deepcopy(base)
    result.update(planned_pairs=30,planned_analyses=60,not_run=60-base['attempted'],
        original_stage_status='USER-STOPPED / INCOMPLETE',sample='RESOURCE-LIMITED TECHNICAL VALIDATION',
        statistical_status='FWER NOT CERTIFIED',scientific_status='SCIENTIFICALLY NOT CERTIFIED')
    for c in result['conditions'].values():
        c['not_run']=30-c['attempted']
        for r in [c['holm_fwer'],c['bh_global_null_fdr'],*c['ordinal'].values()]:
            r['unknown_failure_empirical_bounds']=[r['events']/30,(r['events']+30-r['n'])/30]
    for v in result['contrasts'].values():
        n=v['n'];mean=v['mean'] or 0.;width=4 if len(next(iter(v['table'])))==4 else 2
        v.pop('missing_from_fixed100',None);v['missing_from_fixed30']=30-n
        v['unknown_failure_empirical_bounds']=[(mean*n-width/2*(30-n))/30,(mean*n+width/2*(30-n))/30]
    return result


def verify30(result, rows, masked_ordinal, base):
    A.statistical_audit(rows,masked_ordinal,base)
    assert result['planned_analyses']==60 and result['not_run']==60-len(rows)
    keys={(r['seed'],r['impact']) for r in rows}
    pairs=sum(all((s,k) in keys for k in ('minus','zero')) for s in SEEDS)
    assert result['complete_valid_pairs']==sum(all(next((r['valid'] for r in rows if (r['seed'],r['impact'])==(s,k)),False) for k in ('minus','zero')) for s in SEEDS)
    for c in result['conditions'].values():
        assert c['attempted']+c['not_run']==30
        for rate in [c['holm_fwer'],c['bh_global_null_fdr'],*c['ordinal'].values()]:
            A.close(rate['unknown_failure_empirical_bounds'],[rate['events']/30,(rate['events']+30-rate['n'])/30])
    for v in result['contrasts'].values():
        width=4 if len(next(iter(v['table'])))==4 else 2
        assert v['missing_from_fixed30']==30-v['n']
        total=(v['mean'] or 0.)*v['n'];missing=30-v['n']
        A.close(v['unknown_failure_empirical_bounds'],[(total-width/2*missing)/30,(total+width/2*missing)/30])
    return dict(pairs=pairs,valid_pairs=result['complete_valid_pairs'],independent_statistics_verified=True,
        fixed30_denominators_verified=True,no_optimization=True)


def main():
    start=time.monotonic();out=HERE/'results';snapshot=HERE/'user_stop_snapshot'
    if (out/'resource30_summary.json').exists():raise RuntimeError('Final evidence exists; no overwrite')
    raw=(out/'records.jsonl').read_bytes();rows=[json.loads(line) for line in raw.splitlines()]
    prefix=(snapshot/'records.prefix.jsonl').read_bytes();assert raw.startswith(prefix)
    frozen=json.loads((snapshot/'SHA256.json').read_text())
    for name,digest in frozen.items():assert A.sha((snapshot/name).read_bytes())==digest
    boundary=json.loads((snapshot/'boundary.json').read_text());assert len(rows)==boundary['completed']
    assert len({(r['seed'],r['impact']) for r in rows})==len(rows)
    packed=gzip.compress(raw,mtime=0);assert gzip.decompress(packed)==raw
    (out/'records.jsonl.gz').write_bytes(packed)
    (snapshot/'records.prefix.jsonl.gz').write_bytes(gzip.compress(prefix,mtime=0))
    ordinal=R.ordinal_records()
    R.E.save(out/'ordinal.json',[dict(seed=s,impact=k,data_sha256=v['data_sha256'],methods=v['methods']) for (s,k),v in ordinal.items()])
    original=S.summarize(rows,ordinal);original.update(original_stage_status='USER-STOPPED / INCOMPLETE',
        stop='User resource/quota decision, not statistical success',records_sha256=A.sha(raw),records_gzip_sha256=A.sha(packed))
    R.E.save(out/'summary.json',original)
    independent=A.audit(out);R.E.save(out/'audit.json',independent)
    selected=[r for r in rows if r['seed'] in SEEDS];excess=[r for r in rows if r['seed'] not in SEEDS]
    masked={key:value if key[0] in SEEDS else dict(methods={m:dict(valid=False) for m in ('total','rest')}) for key,value in ordinal.items()}
    base=S.summarize(selected,masked);result=retarget30(base)
    resource_audit=verify30(result,selected,masked,base)
    result['technical_status']='NUMERICALLY VALIDATED — 30 PAIRED SEEDS' if len(selected)==60 and all(r['valid'] for r in selected) and resource_audit['valid_pairs']==30 else 'INCOMPLETE/INVALID — RESOURCE-LIMITED TECHNICAL VALIDATION'
    models=[s for r in selected for s in r['solutions'].values()];calls=[c for r in selected for c in r['optimizer_calls']]
    lr=[v for r in selected for v in r.get('lr',[])]
    history=[json.loads(line) for line in (snapshot/'progress_history.jsonl').read_text().splitlines()]
    approximate_wall=history[-1]['seconds']+rows[-1]['wall_seconds'] if history else sum(r['wall_seconds'] for r in rows)
    result.update(excess_completed=len(excess),excess_keys=[(r['seed'],r['impact']) for r in excess],
        control_initial_completed=json.loads((snapshot/'control.json').read_text())['completed_when_control_installed'],
        native_converged_datasets=sum(len(r['solutions'])==7 and all(s['converged'] for s in r['solutions'].values()) for r in selected),
        completed_hypotheses=sum(r['fits_completed'] for r in selected),native_scipy_successes=sum(c.get('scipy_success',False) for c in calls),
        cold_starts=len(calls),max_GH101_gradient=max(s['native_score'] for s in models),max_GH161_gradient=max(s['audit_score'] for s in models),
        min_native_information_eigenvalue=min(s['native_information_min_eigenvalue'] for s in models),
        min_GH161_information_eigenvalue=min(s['dense_information']['min_eigenvalue'] for s in models),
        max_GH161_information_condition=max(s['dense_information']['condition'] for s in models),
        max_loglik_grid_error=max(s['quadrature_error'] for s in models),max_LR_grid_difference=max(abs(v['audit_minus_fit_lr']) for v in lr),
        analysis_case_wall_seconds=sum(r['wall_seconds'] for r in rows),approximate_stage_wall_seconds_to_boundary=approximate_wall,
        wall_method='last durable59 progress elapsed +60th recorded case wall; excludes subsecond final ledger/control overhead',
        native_fit_seconds=sum(s['fit_seconds'] for s in models),analysis_process_peak_rss_mib=max(r['peak_process_rss_mib'] for r in rows),
        postprocess_seconds=time.monotonic()-start,postprocess_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        independent_audit_seconds=independent['seconds'],total_seconds_approx=approximate_wall+time.monotonic()-start,
        no_completed_refits=True,old_GH61_refits=0,larger_stages_started=False)
    R.E.save(out/'resource30_summary.json',result);R.E.save(out/'resource30_audit.json',resource_audit)
    shutil.copy2('/workspace/scratch/stage-a-execution.txt',out/'execution.txt')
    R.E.save(out/'postprocess_sources.json',{str(p.relative_to(R.ROOT)):A.sha(p.read_bytes()) for p in (Path(__file__),HERE/'user_stop_supervisor.py',HERE/'user_stop_amendment.json')})
    manifest={str(p.relative_to(out)):A.sha(p.read_bytes()) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('SHA256.json','records.jsonl')}
    R.E.save(out/'SHA256.json',manifest)
    print(json.dumps({key:result[key] for key in ('technical_status','statistical_status','scientific_status','attempted','valid','invalid','not_run','complete_valid_pairs','max_GH161_gradient','total_seconds_approx','analysis_process_peak_rss_mib','postprocess_peak_rss_mib')},indent=2))

if __name__=='__main__':main()
