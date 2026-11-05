"""One opt-in preregistered Q4 campaign; cached-evidence audits never fit models."""
import argparse
import base64
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from scipy.stats import beta,t

from validation.binary_se_coverage_study import simulate
from validation.coverage_precision import coverage_precision

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/evidence/p9_1pl_common_se'
PROTOCOL=OUT/'protocol.json'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def versions():return {'python':platform.python_version(),**{n:importlib.metadata.version(n) for n in ('mirt','numpy','scipy','pandas')}}
def atomic(path,value):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as stream:
        json.dump(value,stream,separators=(',',':'),allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)


def fit_one(task):
    index,state=task;state=Path(state);result=state/f'{index:05d}.json';claim=state/f'{index:05d}.claim'
    if result.exists():return index
    if claim.exists():
        atomic(result,dict(index=index,status='INTERRUPTED',converged=False,estimate=[None]*13,se=[None]*13,
                           reason='Previously claimed but uncollected; not refitted',seconds=0,data_sha256=None))
        return index
    with claim.open('x') as f:f.write(str(index))
    tick=time.monotonic();p=json.loads(PROTOCOL.read_text());record=dict(index=index,status='ERROR',converged=False,
        estimate=[None]*13,se=[None]*13,data_sha256=None)
    try:
        from core.irt_python_mirt_adapter import fit_python_mirt
        from core.irt_common_a_inference import parameter_inference,common_a_basis
        truth=np.asarray(p['DGP']['b']);rng=np.random.default_rng(np.random.SeedSequence([20261029,4,index]))
        x=simulate(rng,np.full(12,p['DGP']['a_common']),truth,p['N'])
        record['data_sha256']=hashlib.sha256(x.astype('uint8').tobytes()).hexdigest()
        if ((x.mean(0)==0)|(x.mean(0)==1)).any():raise ValueError('CONSTANT ITEM')
        fit=fit_python_mirt(x,model='1PL',common_discrimination=True).fit
        record.update(converged=bool(fit.converged),iterations=int(fit.n_iterations),
                      loglik=float(fit.log_likelihood) if np.isfinite(fit.log_likelihood) else None,
                      se_method=str(fit.se_method),n_parameters=int(fit.n_parameters))
        co=fit.coef();est=np.r_[co.discrimination.iloc[0],co.difficulty.to_numpy()]
        record['estimate']=[float(v) if np.isfinite(v) else None for v in est]
        if not fit.converged:
            record.update(status='NONCONVERGED',reason='Library convergence false')
        else:
            if fit.se_method!='oakes':raise ValueError('OAKES OBSERVED INFORMATION METHOD NOT VERIFIED')
            if fit.latent_mean is not None and not np.array_equal(fit.latent_mean,np.zeros(1)):
                raise ValueError('LATENT MEAN MISMATCH')
            if fit.latent_covariance is not None and not np.array_equal(fit.latent_covariance,np.eye(1)):
                raise ValueError('LATENT VARIANCE MISMATCH')
            table,why=parameter_inference(fit,'1PL',[f'M{j}' for j in range(12)])
            if table is None:record.update(status='CI_WITHHELD',reason=why)
            else:
                a=table[table.Parametre=='a'];b=table[table.Parametre=='b']
                if len(a)!=1 or len(b)!=12:raise ValueError('COMMON PARAMETER COUNT MISMATCH')
                se=np.r_[a.SH.to_numpy(),b.SH.to_numpy()]
                record['se']=[float(v) if np.isfinite(v) else None for v in se]
                cov,_=common_a_basis(fit);eig=np.linalg.eigvalsh((cov+cov.T)/2)
                record.update(status='OK',covariance_min_eigenvalue=float(eig[0]),
                              covariance_condition=float(eig[-1]/eig[0]),reason='')
                if index<3:record['independent_covariance']=cov.tolist()
    except Exception as exc:record.update(reason=str(exc),status='ERROR')
    record['seconds']=time.monotonic()-tick;atomic(result,record)
    return index


def cp(k,n,alpha=.05):
    if not n:return [None,None]
    return [float(beta.ppf(alpha/2,k,n-k+1)) if k else 0.,
            float(beta.ppf(1-alpha/2,k+1,n-k)) if k<n else 1.]


def summarize(rows,protocol):
    r=protocol['R'];truth=np.r_[protocol['DGP']['a_common'],protocol['DGP']['b']]
    est=np.asarray([v['estimate'] for v in rows],float);se=np.asarray([v['se'] for v in rows],float)
    convergence=np.asarray([v['converged'] for v in rows],bool)
    eligible=convergence[:,None]&np.isfinite(est)&np.isfinite(se)&(se>0)
    covered=eligible&(est-1.959963984540054*se<=truth)&(truth<=est+1.959963984540054*se)
    point=convergence[:,None]&np.isfinite(est)
    checks={'complete':len(rows)==r,'unique_indices':sorted(v['index'] for v in rows)==list(range(r)),
            'convergence':float(convergence.sum()/r)>=.99}
    result=dict(attempted=len(rows),planned=r,converged=int(convergence.sum()),
        convergence_rate=float(convergence.sum()/r),statuses=dict(Counter(v['status'] for v in rows)),
        reasons=dict(Counter(v.get('reason','') for v in rows if v['status']!='OK')),parameters={})
    for j,name in enumerate(['a_common']+[f'b{j+1:02d}' for j in range(12)]):
        n=int(eligible[:,j].sum());k=int(covered[:,j].sum());rate=k/n if n else None
        precision=coverage_precision([np.c_[1-eligible[i,j:j+1],covered[i,j:j+1]] for i in range(len(rows))],r,eligible_column=0)
        mcse=precision['coverage_mc_se'];error=est[point[:,j],j]-truth[j]
        sensitivity=[k/r,(k+r-n)/r]
        cell=dict(truth=float(truth[j]),eligible=n,covered=k,coverage=rate,valid_CI_proportion=n/r,
            coverage_mcse=mcse,independent_replications=r,
            effective_sample_size=float(rate*(1-rate)/mcse**2) if mcse and rate is not None else None,
            cp95=cp(k,n),simultaneous_cp95=cp(k,n,.05/13),missing_outcome_bounds=sensitivity,
            mean_bias=float(error.mean()) if len(error) else None,
            bias_mcse=float(error.std(ddof=1)/np.sqrt(len(error))) if len(error)>1 else None,
            rmse=float(np.sqrt(np.mean(error**2))) if len(error) else None,
            mean_SE=float(np.mean(se[eligible[:,j],j])) if n else None,
            empirical_SD=float(est[point[:,j],j].std(ddof=1)) if len(error)>1 else None)
        cell['SE_to_SD']=cell['mean_SE']/cell['empirical_SD'] if cell['mean_SE'] and cell['empirical_SD'] else None
        result['parameters'][name]=cell
        checks[name+'_bias']=cell['mean_bias'] is not None and abs(cell['mean_bias'])<=.05
        checks[name+'_coverage']=rate is not None and abs(rate-.95)<=.02
        checks[name+'_precision']=mcse is not None and mcse<=.004 and n>=3600
        checks[name+'_valid']=n/r>=.95
        checks[name+'_failure_sensitivity']=sensitivity[0]>=.93 and sensitivity[1]<=.97
    def pooled(indices):
        e=eligible[:,indices];c=covered[:,indices]
        parts=[np.c_[1-e[i],c[i]] for i in range(len(rows))]
        precision=coverage_precision(parts,r,eligible_column=0);n=int(e.sum());k=int(c.sum())
        rate=k/n if n else None;mcse=precision['coverage_mc_se']
        interval=[max(0.,rate-float(t.ppf(.975,r-1))*mcse),min(1.,rate+float(t.ppf(.975,r-1))*mcse)] if rate is not None and mcse is not None else [None,None]
        return dict(eligible=n,covered=k,coverage=rate,coverage_mcse=mcse,cluster_t95=interval,
            effective_sample_size=rate*(1-rate)/mcse**2 if mcse else None,
            independent_replications=r,eligible_replications=precision['eligible_replications'],
            mcse_method=precision['coverage_mc_se_method'])
    result['pooled_b']=pooled(list(range(1,13)));result['pooled_all_descriptive']=pooled(list(range(13)))
    b=result['pooled_b'];checks['pooled_b_coverage']=b['coverage'] is not None and abs(b['coverage']-.95)<=.02
    checks['pooled_b_precision']=b['coverage_mcse'] is not None and b['coverage_mcse']<=.004 and b['eligible']>=3600
    result['checks']=checks;result['status']='CERTIFIED' if all(checks.values()) else 'NOT CERTIFIED'
    return result


def read_rows(folder=OUT):
    manifest=json.loads((folder/'raw_manifest.json').read_text());encoded=''
    for part in manifest['parts']:
        path=folder/part['path']
        if digest(path)!=part['sha256']:raise ValueError('Raw part identity mismatch')
        encoded+=path.read_text()
    packed=base64.b64decode(encoded)
    if hashlib.sha256(packed).hexdigest()!=manifest['gzip_sha256']:raise ValueError('Raw gzip identity mismatch')
    raw=gzip.decompress(packed)
    if hashlib.sha256(raw).hexdigest()!=manifest['raw_sha256']:raise ValueError('Raw record identity mismatch')
    return [json.loads(v) for v in raw.splitlines()]


def run(state,protocol_commit,workers=2):
    protocol=json.loads(PROTOCOL.read_text());state=Path(state);state.mkdir(parents=True,exist_ok=True)
    source={**protocol['source_sha256'],'validation/common_a_se_study.py':digest(__file__)}
    if any(digest(ROOT/p)!=h for p,h in source.items()):raise ValueError('Preregistered source changed')
    config=dict(protocol_commit=protocol_commit,protocol_sha256=digest(PROTOCOL),versions=versions(),source_sha256=source)
    if (state/'config.json').exists():
        if json.loads((state/'config.json').read_text())!=config:raise ValueError('Checkpoint configuration mismatch')
    else:atomic(state/'config.json',config)
    started=time.monotonic();tasks=[(i,str(state)) for i in range(protocol['R']) if not(state/f'{i:05d}.json').exists()]
    print(json.dumps({'new_tasks':len(tasks),'planned':protocol['R'],'workers':workers}),flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(fit_one,task) for task in tasks]
        for count,future in enumerate(as_completed(fs),1):
            future.result()
            if count%250==0:print(json.dumps({'completed_new':count,'seconds':round(time.monotonic()-started,1)}),flush=True)
    rows=[json.loads((state/f'{i:05d}.json').read_text()) for i in range(protocol['R'])]
    raw=('\n'.join(json.dumps(v,separators=(',',':'),allow_nan=False) for v in rows)+'\n').encode()
    packed=gzip.compress(raw,mtime=0);encoded=base64.b64encode(packed).decode();parts=[]
    for i,start in enumerate(range(0,len(encoded),140000)):
        path=OUT/f'raw_{i:02d}.b64';path.write_text(encoded[start:start+140000]);parts.append({'path':path.name,'sha256':digest(path)})
    atomic(OUT/'raw_manifest.json',dict(parts=parts,records=len(rows),gzip_sha256=hashlib.sha256(packed).hexdigest(),raw_sha256=hashlib.sha256(raw).hexdigest()))
    result=summarize(rows,protocol);result.update(config,wall_seconds=time.monotonic()-started,
        scope='Only N1000 J12, true common a1, specified b grid, normal theta; ML/GH21/tol1e-4/max_iter500. No extrapolation.',
        certification_condition='Scientific criteria plus required CI for this exact source revision; no general guarantee.',
        campaigns=1,other_Monte_Carlo_campaigns=0)
    atomic(OUT/'coverage.json',result)
    print(json.dumps({'status':result['status'],'converged':result['converged'],'checks_failed':[k for k,v in result['checks'].items() if not v]}),flush=True)


def ci_mode(base):
    allowed={'.github/workflows/mtk-scientific-gates.yml','.github/workflows/pytest.yml',
        'core/irt_common_a_inference.py','core/irt_python_mirt_report.py','ui/irt_python_library_view.py',
        'tests/test_common_a_se.py','validation/common_a_se_study.py',
        'validation/CODEX_CERTIFICATION_QUEUE.md','validation/CERTIFICATION_PREAUDIT.md',
        'validation/SCIENTIFIC_MODEL_CERTIFICATION_MATRIX.md'}
    paths=subprocess.check_output(['git','diff','--name-only',base,'HEAD'],text=True).splitlines()
    if not paths or any(p not in allowed and not p.startswith('validation/evidence/p9_1pl_common_se/') for p in paths):
        raise ValueError('Q4 archive-only CI rejected: unrelated/engine changes')
    protocol=json.loads(PROTOCOL.read_text())
    if any(digest(ROOT/p)!=h for p,h in protocol['source_sha256'].items()):raise ValueError('Campaign source identity changed')
    with open(os.environ['GITHUB_OUTPUT'],'a') as stream:stream.write('evidence_only=true\n')
    print('Only the already-recorded Q4 campaign; no Monte Carlo in CI; all evidence/browser checks retained.')


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--run',action='store_true');g.add_argument('--audit',action='store_true');g.add_argument('--ci-mode',action='store_true')
    p.add_argument('--state',type=Path);p.add_argument('--protocol-commit');p.add_argument('--workers',type=int,default=2);p.add_argument('--base')
    a=p.parse_args()
    if a.run:
        if not a.state or not a.protocol_commit:p.error('Run requires durable --state and --protocol-commit')
        run(a.state,a.protocol_commit,a.workers)
    elif a.ci_mode:ci_mode(a.base)
    else:
        d=summarize(read_rows(),json.loads(PROTOCOL.read_text()));saved=json.loads((OUT/'coverage.json').read_text())
        if d['checks']!=saved['checks'] or d['status']!=saved['status']:raise ValueError('Archived scientific decision mismatch')
        print(json.dumps({'status':d['status'],'records':d['attempted'],'new_fits':0}))
