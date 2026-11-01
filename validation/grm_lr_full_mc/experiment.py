"""Durable staged continuation. Reuses PR122 fits; never retries a failed fit."""
import gzip
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from validation.ordinal_dif_grm_lr_paired import experiment as F
from validation.ordinal_dif_impact_zero import experiment as Z
from validation.ordinal_dif_paired import experiment as P
from validation.dif_validation_study import gen_ordinal

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PILOT=F.HERE/'results'


def protect():
    m=json.loads((HERE/'prior_manifest.json').read_text())
    for p,h in m.items():
        if Z.digest((ROOT/p).read_bytes())!=h:raise ValueError('Prior source/evidence changed: '+p)
    return len(m)


def cached_records():
    return {k:{r['seed']:r for r in Z.load_records(p/'results/records.jsonl.gz')} for k,p in (('minus',Z.OLD),('zero',Z.HERE))}


def pilot_records():
    return json.loads(gzip.decompress((PILOT/'records.json.gz').read_bytes()))['records']


def stream_new(output):
    for path in sorted(output.glob('chunk_*.jsonl.gz')):
        for line in gzip.decompress(path.read_bytes()).splitlines():yield json.loads(line)
    if (output/'active.jsonl').exists():
        for line in (output/'active.jsonl').read_bytes().splitlines():yield json.loads(line)


def slim(r):
    result={k:v for k,v in r['result'].items() if k not in ('fits','table','gate_audit')}
    return {**{k:v for k,v in r.items() if k not in ('result','crn','stdout','stderr')},'result':result}


class Ledger:
    def __init__(self,output):
        self.output=output;self.path=output/'active.jsonl';self.count=0
    def append(self,r):
        with self.path.open('ab') as file:
            file.write((json.dumps(P.clean(r),allow_nan=False)+'\n').encode());file.flush();os.fsync(file.fileno())
        self.count+=1
        if self.count>=100:self.seal()
    def seal(self):
        if not self.path.exists():return
        raw=self.path.read_bytes();packed=gzip.compress(raw,mtime=0)
        assert gzip.decompress(packed)==raw
        i=len(list(self.output.glob('chunk_*.jsonl.gz')))
        target=self.output/f'chunk_{i:03d}.jsonl.gz';target.write_bytes(packed)
        self.path.unlink();self.count=0


def summary(rows,cached,used,stop,stage):
    s=F.summarize(rows,cached,used,stop)
    for key in ('feasible_for_staged_future_study','main_within_current_600s_budget','main_executed'):
        s.pop(key,None)
    counts={k:sum(r['impact']==k for r in rows) for k in ('minus','zero')}
    by={(r['seed'],r['impact']):r for r in rows}
    paired=sum(all((seed,k) in by for k in ('minus','zero')) for seed in {r['seed'] for r in rows})
    complete=all(v==2500 for v in counts.values()) and s['valid']==5000
    s.update(stage=stage,completed_attempt_pairs=paired,required_pairs=2500,not_attempted={k:2500-v for k,v in counts.items()},fixed_design_complete=complete,
        scientific_status='MEASURED / NOT CERTIFIED' if complete else 'INCOMPLETE / NOT CERTIFIED',
        inference_note='Confirmatory conclusions disabled unless fixed2500 all-valid pairs. Valid-case rates/intervals after technical stop are conditional descriptions, not nominal FWER certification.',
        pilot_reused_analyses=20,new_attempted=len(rows)-20,budget_seconds=18000,prior_files_verified=protect())
    for k,c in s['conditions'].items():
        n=c['valid'];p=c['fwer'];c['scoped_joint_decision']='INCONCLUSIVE / INCOMPLETE'
        if complete:
            lo,hi=c['joint_nine_cp'];c['scoped_joint_decision']='SCOPED INFLATION' if lo>.05 else 'SCOPED CONTROL' if hi<=.05 else 'INCONCLUSIVE'
    return s


def finalize(output,cached,used,stop,stage):
    rows=[slim(r) for r in pilot_records()]+[slim(r) for r in stream_new(output)]
    keys=[(r['seed'],r['impact']) for r in rows]
    if len(set(keys))!=len(keys):raise ValueError('Repeated seed/condition: no silent duplication')
    s=summary(rows,cached,used,stop,stage)
    P.save(output/'summary.json',s)
    archives={p.name:dict(sha256=Z.digest(p.read_bytes()),bytes=p.stat().st_size) for p in sorted(output.glob('chunk_*.jsonl.gz'))}
    P.save(output/'archive_manifest.json',archives)
    return s


def run():
    output=HERE/'results';output.mkdir(exist_ok=True)
    if (output/'metadata.json').exists():raise RuntimeError('Existing evidence protected: do not repeat attempts; use audit/finalize, not a fresh fit run')
    protect();cached=cached_records();pilot=pilot_records()
    if len(pilot)!=20 or not all(r['result']['valid'] and F.gates(r['result'])['valid'] for r in pilot):raise ValueError('Pilot invalid or incomplete')
    oldmeta=json.loads((PILOT/'metadata.json').read_text())
    versions={p:importlib.metadata.version(p) for p in oldmeta['versions']}
    if versions!=oldmeta['versions']:raise ValueError('Package versions differ from pilot')
    for p,h in oldmeta['sources'].items():
        if Z.digest((ROOT/p).read_bytes())!=h:raise ValueError('Pilot source changed: '+p)
    traces={r['seed']:r for r in [json.loads(line) for line in gzip.decompress((Z.HERE/'results/crn.jsonl.gz').read_bytes()).splitlines()]}
    plan=json.loads((HERE/'plan.json').read_text());pilot_seconds=json.loads((PILOT/'summary.json').read_text())['seconds']
    sources=[str(p.relative_to(ROOT)) for p in (HERE/'experiment.py',HERE/'plan.json',HERE/'prior_manifest.json',HERE/'audit.py')]
    P.save(output/'metadata.json',dict(versions=versions,python=sys.version,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={**oldmeta['sources'],**{p:Z.digest((ROOT/p).read_bytes()) for p in sources}},plan=plan,pilot_records_sha256=Z.digest((PILOT/'records.json.gz').read_bytes()),pilot_seconds=pilot_seconds))
    rows=[slim(r) for r in pilot];ledger=Ledger(output);start=time.monotonic();stop='fixed design complete';stage=10
    def used():return pilot_seconds+time.monotonic()-start
    def state():P.save(output/'progress.json',dict(used_seconds=used(),attempted=len(rows),valid=sum(r['result'].get('valid',False) for r in rows),stage=stage,stop=stop))
    try:
        with tempfile.TemporaryDirectory(prefix='grm-main-') as directory:
            tmp=Path(directory)
            for specification in plan['stages']:
                target=specification['pairs'];stage_start=time.monotonic();previous=stage
                if used()/len(rows)*(5000-len(rows))>18000-used():stop='projected remaining cost exceeds budget';break
                for index in range(previous,target):
                    seed=2026110100+index
                    zero,g,trace=Z.paired_control(seed,cached['minus'][seed]);minus,gm=gen_ordinal(500,500,-.5,False,seed)
                    if trace!=traces[seed] or not np.array_equal(g,gm):raise ValueError('Common-randomness replay differs')
                    for impact,data in (('minus',minus),('zero',zero)):
                        if Z.data_hash(data,g)!=cached[impact][seed]['data_sha256']:raise ValueError('Archived response hash mismatch')
                        remaining=min(18000-used(),specification['increment_wall_cap_seconds']-(time.monotonic()-stage_start))
                        if remaining<=0:stop='wall budget exhausted';break
                        frame=data.copy();frame['group']=g;source=tmp/'data.csv';frame.to_csv(source,index=False);destination=tmp/'result.json';destination.unlink(missing_ok=True)
                        marker=dict(seed=seed,index=index,impact=impact,data_sha256=Z.data_hash(data,g),utc_start=time.time(),stage_target=target)
                        P.save(output/'attempt_started.json',marker)
                        t=time.monotonic();process=subprocess.Popen([sys.executable,'-m','validation.ordinal_dif_grm_lr_paired.experiment','--worker',str(source),str(destination)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                        timeout=False
                        try:stdout,stderr=process.communicate(timeout=min(90,remaining))
                        except subprocess.TimeoutExpired:
                            timeout=True;process.kill();stdout,stderr=process.communicate()
                        result=json.loads(destination.read_text()) if destination.exists() else dict(valid=False,reason='timeout' if timeout else 'worker output missing')
                        if timeout or process.returncode:result.update(valid=False,worker_error=True)
                        fits=[f for item in result.get('fits',{}).values() for f in item.values()]
                        record=dict(seed=seed,index=index,impact=impact,data_sha256=marker['data_sha256'],pairing_trace_sha256=Z.digest(json.dumps(trace,sort_keys=True).encode()),
                            result=result,wall_seconds=time.monotonic()-t,returncode=process.returncode,timeout=timeout,
                            all_native_converged=bool(len(fits)==8 and all(f['converged'] for f in fits)),stdout=stdout,stderr=stderr)
                        if result.get('valid',False) and not F.gates(result)['valid']:result.update(valid=False,reason='Native result/gate audit inconsistent')
                        if not result.get('valid',False):
                            path=output/f'failed_{index:04d}_{impact}.npz';np.savez_compressed(path,X=data.to_numpy(dtype=np.uint8),group=g.astype(np.uint8));record['failed_matrix']=path.name;record['matrix_sha256']=Z.digest(path.read_bytes())
                        ledger.append(record);rows.append(slim(record));(output/'attempt_started.json').unlink();state()
                        if not result.get('valid',False):stop='scientific/native/worker gate failed';break
                        if result.get('peak_rss_mib',0)>1024:stop='memory safety threshold';break
                        if len(rows)%20==0:print(f'{len(rows)}/5000 analyses completed; {used():.1f}s including pilot; stage target {target}',flush=True)
                    if stop!='fixed design complete':break
                if stop!='fixed design complete':break
                stage=target;ledger.seal();state()
                # Technical boundary audit only; no event estimates used for continuation.
                if protect()!=len(json.loads((HERE/'prior_manifest.json').read_text())):raise ValueError('Evidence protection mismatch')
                loaded=[slim(r) for r in pilot]+[slim(r) for r in stream_new(output)]
                if len(loaded)!=2*target or not all(r['result']['valid'] for r in loaded):raise ValueError('Incomplete/invalid stage')
                for r in loaded:
                    if r['data_sha256']!=cached[r['impact']][r['seed']]['data_sha256']:raise ValueError('Stage pairing hash mismatch')
                P.save(output/f'stage_{target:04d}.json',dict(completed_pairs=target,analyses=len(rows),valid=len(rows),used_seconds=used(),peak_worker_rss_mib=max(r['result'].get('peak_rss_mib',0) for r in rows),projected_total_seconds=used()/len(rows)*5000,prior_files_verified=protect(),decision_basis='technical gates/runtime only; no statistical success decision'))
                print(f'Stage {target} technical audit complete; {used():.1f}s',flush=True)
    except Exception as exc:
        stop=f'BLOCKED {type(exc).__name__}: {exc}'
        P.save(output/'exception.json',dict(reason=stop))
    finally:
        marker_path=output/'attempt_started.json'
        if marker_path.exists():
            pending=json.loads(marker_path.read_text())
            if (pending['seed'],pending['impact']) not in {(r['seed'],r['impact']) for r in rows}:
                interrupted={**pending,'result':dict(valid=False,reason=stop),'wall_seconds':0.,'all_native_converged':False,'interrupted':True}
                ledger.append(interrupted);rows.append(slim(interrupted))
            marker_path.unlink()
        ledger.seal();state()
    s=finalize(output,cached,used(),stop,stage)
    from validation.grm_lr_full_mc.audit import audit
    P.save(output/'audit.json',audit(output))
    print(json.dumps({k:v for k,v in s.items() if k!='contrasts'},indent=2),flush=True)


if __name__=='__main__':run()
