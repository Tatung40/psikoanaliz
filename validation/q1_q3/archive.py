"""Archive actual R artefacts and deterministic Python replays; never run MC or R."""
import argparse
import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import zlib

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'validation/evidence'


def sha(b):return hashlib.sha256(b).hexdigest()
def pack(path):
    b=Path(path).read_bytes()
    return {'sha256':sha(b),'zlib_base64':base64.b64encode(zlib.compress(b,9)).decode()}
def unpack(blob):
    b=zlib.decompress(base64.b64decode(blob['zlib_base64']))
    if sha(b)!=blob['sha256']:raise ValueError('Archived raw evidence corrupted')
    return b
def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value))
def metadata(run,artifact,checkout):
    return dict(run=run,url=f'https://github.com/Tatung40/psikoanaliz/actions/runs/{run}',
                artifact=artifact,checkout=checkout,
                replay_versions={'python':platform.python_version(),**{n:importlib.metadata.version(n)
                    for n in ('mirt','numpy','scipy','pandas')}},
                independent_R_execution='ARCHIVED ACTUAL EXECUTION; no new R fit',
                monte_carlo_executions=0,certified=False)
def save(name,d):
    (OUT/name).write_text(json.dumps(d,ensure_ascii=False,default=serial,separators=(',',':'),allow_nan=False)+'\n')


def poly(folder):
    from validation.r_mirt_polytomous.compare import (
        python_fit,from_r,probabilities,common_loglik,THETA)
    root=Path(folder)/'r-mirt-polytomous'
    criteria=json.loads((ROOT/'validation/mirt_stage1/criteria.json').read_text())
    result=metadata(40957313137,12933083519,'7fa43e84a6a2b4106ed28a75b763514e74fe39c6')
    result.update(criteria=criteria,original_report=pack(root/'report.json'),cases=[])
    for model in ('GRM','PCM','GPCM','NRM'):
        d=root/model;data=pd.read_csv(d/'data.csv');x=data.to_numpy()
        r=json.loads((d/'r_result.json').read_text())
        fit,py,eap,psd=python_fit(model,x)
        rp=from_r(model,r['coef']);reap=np.asarray(r['eap'])
        reflected=model=='NRM' and np.corrcoef(eap,reap)[0,1]<0
        if reflected:py={'slopes':-py['slopes'],'intercepts':py['intercepts']};eap=-eap
        names=[k for k in py if not(model=='PCM' and k=='discrimination')]
        diff=dict(mapping_probability=float(np.abs(probabilities(model,rp,THETA)-np.asarray(r['probtrace'])).max()),
                  calibration_parameters=max(float(np.abs(py[k]-rp[k]).max()) for k in names),
                  calibration_probability=float(np.abs(probabilities(model,py,THETA)-probabilities(model,rp,THETA)).max()),
                  common_logLik=abs(common_loglik(model,py,x)-common_loglik(model,rp,x)),
                  calibration_EAP=float(np.abs(eap-reap).max()),
                  calibration_posterior_SD=float(np.abs(psd-np.asarray(r['eap_se'])).max()))
        checks={k:v<=criteria[k] for k,v in diff.items()}
        checks.update(converged=bool(fit.converged and r['converged']),free_parameters=fit.n_parameters==r['nest'])
        result['cases'].append(dict(model=model,data=pack(d/'data.csv'),R_raw=pack(d/'r_result.json'),
            R_versions={'R':r['R'],'mirt':r['mirt']},python_parameters=py,R_parameters=rp,
            Python_converged=bool(fit.converged),R_converged=r['converged'],
            logLik={'python_native':float(fit.log_likelihood),'R_native':r['loglik'],
                    'python_common_GH121':common_loglik(model,py,x),'R_common_GH121':common_loglik(model,rp,x)},
            python_reflected=bool(reflected),differences=diff,checks=checks,
            settings={'estimation':'ML, no priors, N(0,1)','python':'GH61; tol1e-8; max_iter5000',
                      'R':'101 regular nodes [-6,6]; TOL1e-8; see frozen reference.R',
                      'comparison':'Common GH121 likelihood and81 display nodes'},
            boundaries='Not independently recorded by original comparison; do not certify stationarity/bounds',
            status='NUMERICAL AGREEMENT / NOT CERTIFIED' if all(checks.values()) else 'FAIL'))
        print('archived',model,result['cases'][-1]['status'],flush=True)
    save('r_polytomous.json',result)


def binary_and_rsm(folder):
    from validation.r_mirt_34.compare import python_fit,params,irf,information,common_loglik,THETA,SE_RELATIVE_MAX
    root=Path(folder);criteria=json.loads((ROOT/'validation/mirt_stage1/criteria.json').read_text())
    result=metadata(41226545056,13051448638,'a85d32af6d4a29d26fe8e50382a169b8e122446a')
    result.update(criteria=criteria,se_relative_max=SE_RELATIVE_MAX,
                  original_report=pack(root/'r-mirt-34/report.json'),cases=[])
    d=root/'r-mirt-34/3PL';data=pd.read_csv(d/'data.csv');x=data.to_numpy();r=json.loads((d/'r_result.json').read_text())
    fit,table,reason=python_fit('3PL',x);assert table is not None,reason
    py=params(table,'3PL');rp={k:np.asarray(v) for k,v in r['par'].items()}
    a,b=information(py,THETA),information(rp,THETA);tif=b.sum(1)
    pyse={q:table[table.Parametre==q].SH.to_numpy() for q in 'abc'}
    differences=dict(calibration_parameters=max(float(np.abs(py[q]-rp[q]).max()) for q in 'abc'),
        calibration_probability=float(np.abs(irf(py,THETA)-irf(rp,THETA)).max()),
        common_logLik=abs(common_loglik(py,x)-common_loglik(rp,x)),
        information_relative=float(np.abs(a-b).max()/tif.max()),
        tif_relative=float(np.abs(a.sum(1)-tif).max()/tif.max()),
        se_relative=max(float(np.abs(pyse[q]/np.asarray(r['se'][q])-1).max()) for q in 'abc'))
    checks={k:differences[k]<=criteria[k] for k in ('calibration_parameters','calibration_probability','common_logLik')}
    checks.update(information=differences['information_relative']<=criteria['calibration_probability'],
                  tif=differences['tif_relative']<=criteria['calibration_probability'],
                  se=differences['se_relative']<=SE_RELATIVE_MAX,
                  converged=bool(fit.converged and r['converged']),python_interior=bool((table.Not=='').all()))
    result['cases'].append(dict(model='3PL',seed=13,data=pack(d/'data.csv'),R_raw=pack(d/'r_result.json'),
        R_versions={'R':r['R'],'mirt':r['mirt']},python_parameters=py,R_parameters=rp,
        python_SE=pyse,R_SE=r['se'],differences=differences,checks=checks,
        python_native_logLik=float(fit.log_likelihood),R_native_logLik=r['loglik'],
        status='NUMERICAL AGREEMENT / NOT CERTIFIED' if all(checks.values()) else 'FAIL'))
    result['cases'].append(json.loads((root/'r-mirt-34/report.json').read_text())[1])
    result['limitation']='3PL interior-selected fixture only; native quadratures differ. 4PL NOT VERIFIED is preserved, not PASS.'
    save('r_34.json',result);print('archived 3PL',result['cases'][0]['status'],flush=True)
    from core import irt_rsm as rsm
    result=metadata(41226545056,13051448638,'a85d32af6d4a29d26fe8e50382a169b8e122446a')
    result.update(criteria=criteria,original_report=pack(root/'r-mirt-rsm/report.json'),cases=[])
    for case in ('irt_027','irt_028','irt_029'):
        d=root/'r-mirt-rsm'/case;data=pd.read_csv(d/'data.csv');r=json.loads((d/'r_result.json').read_text())
        f=rsm.fit(data,points=61);trace=np.asarray(r['probtrace'])
        # Exact RSM adjacent-category log-odds at theta0: step=-log(Pc/Pc-1).
        steps=-np.log(trace[:,40,1:]/trace[:,40,:-1])
        probability=np.exp(rsm.category_log_probs(THETA,f['steps'])).transpose(1,0,2)
        differences={'calibration_parameters':abs(f['sd']-r['sd']),
                     'calibration_probability':float(np.abs(probability-trace).max()),
                     'common_logLik':abs(f['loglik']-r['loglik'])}
        checks={k:v<=criteria[k] for k,v in differences.items()}
        checks.update(converged=bool(f['converged'] and r['converged']),free_parameters=f['npar']==r['nest'])
        result['cases'].append(dict(case=case,data=pack(d/'data.csv'),R_raw=pack(d/'r_result.json'),
            R_versions={'R':r['R'],'mirt':r['mirt']},python_steps=f['steps'],R_steps_from_trace=steps,
            step_difference=float(np.abs(f['steps']-steps).max()),python_sd=f['sd'],R_sd=r['sd'],
            python_logLik=f['loglik'],R_logLik=r['loglik'],differences=differences,checks=checks,
            settings='Rasch a1 fixed; mean0; SD free. NativeGH61 vs R regular101, not common-grid logLik despite historical field name.',
            boundaries='Original reference did not retain gradient/boundary audit; not certified.',
            status='NUMERICAL AGREEMENT / NOT CERTIFIED' if all(checks.values()) else 'FAIL'))
        print('archived',case,result['cases'][-1]['status'],flush=True)
    save('r_rsm.json',result)


def gt(folder):
    root=Path(folder)/'gtheory-r-diagnostics'
    result=metadata(42013874602,13407453866,'b72d6d4ae19de64dface769369e1de0490a009a9')
    result['archive_process_versions']=result.pop('replay_versions')
    result['python_estimates_origin']='Original actual CI CSVs, no new G-Theory replay. GeneralizIT0.1.2 recorded per comparison; original Python patch/NumPy/pandas versions were not retained in this artifact.'
    result.update(criteria={'anova':{'rtol':1e-10,'atol':1e-10},'rounded_variance':{'rtol':0,'atol':1e-12},
                            'error_variance':{'rtol':1e-12,'atol':1e-12},'rounded_coefficients':'exact'},cases=[])
    for n in range(5):
        d=root/f'test_independent_r_anova_and_g{n}'
        comparison=json.loads((d/'comparison.json').read_text())
        assert comparison['fixture_sha256']==sha((ROOT/'validation/generalizability/balanced.csv').read_bytes())
        paths=[p for p in d.rglob('*') if p.is_file()]
        raw={str(p.relative_to(d)):pack(p) for p in sorted(paths)}
        tables={str(p.relative_to(d)):pd.read_csv(p).astype(object).where(pd.notna(pd.read_csv(p)),None).to_dict('records')
                for p in d.rglob('*.csv')}
        result['cases'].append(dict(comparison=comparison,raw=raw,tables=tables,
            R_session=(d/'r/sessionInfo.txt').read_text(),
            settings=json.loads((d/'python_settings.json').read_text()),
            scope='Five existing balanced random-facet deterministic fixtures only; no new GT design or inference.'))
    save('r_gt.json',result)


def matched(folder):
    root=Path(folder)/'refit';ref=root/'4PL/R_96wsqzr8'
    d=json.loads((OUT/'r_34.json').read_text())
    d['historical_matched_4PL']=dict(run=40151828898,artifact=12582966948,
        status='FAIL / R NOT CONVERGED / NOT CERTIFIED',report=pack(root/'report.json'),
        Python_parameters=pack(root/'4PL/python_parameters.csv'),R_parameters=pack(ref/'parameters.csv'),
        integration=pack(ref/'integration.csv'),R_curves=pack(ref/'curves.csv'),
        R_metadata=pack(ref/'metadata.csv'),R_session=pack(ref/'sessionInfo.txt'),
        R_stderr=pack(root/'4PL/R_stderr.txt'),R_stdout=pack(root/'4PL/R_stdout.txt'),
        fixture='tests/fixtures/irt_b11/data_irt4pl_ok.csv',fixture_sha256=sha((root/'4PL/data.csv').read_bytes()))
    save('r_34.json',d)


def restore(folder):
    """Restore committed original bytes; no fitting, simulation or downloads."""
    root=Path(folder)
    def write(path,blob):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(unpack(blob))
    for file,base in [('r_polytomous.json','3/r-mirt-polytomous'),('r_rsm.json','0/r-mirt-rsm'),('r_34.json','0/r-mirt-34')]:
        d=json.loads((OUT/file).read_text());write(root/base/'report.json',d['original_report'])
        for case in d['cases']:
            if 'data' not in case:continue
            path=root/base/case.get('model',case.get('case'))
            write(path/'data.csv',case['data']);write(path/'r_result.json',case['R_raw'])
            np.savetxt(path/'theta.csv',np.linspace(-4,4,81),delimiter=',',header='theta',comments='')
    d=json.loads((OUT/'r_gt.json').read_text())
    for n,case in enumerate(d['cases']):
        for name,blob in case['raw'].items():write(root/'1/gtheory-r-diagnostics'/f'test_independent_r_anova_and_g{n}'/name,blob)
    raw=json.loads((OUT/'r_34.json').read_text())['historical_matched_4PL']
    base=root/'2/refit';ref=base/'4PL/R_96wsqzr8'
    write(base/'report.json',raw['report']);write(base/'4PL/python_parameters.csv',raw['Python_parameters'])
    (base/'4PL/data.csv').write_bytes((ROOT/raw['fixture']).read_bytes())
    for key,name in [('R_parameters','parameters.csv'),('integration','integration.csv'),('R_curves','curves.csv'),
                     ('R_metadata','metadata.csv'),('R_session','sessionInfo.txt')]:write(ref/name,raw[key])
    for key,name in [('R_stdout','R_stdout.txt'),('R_stderr','R_stderr.txt')]:write(base/'4PL'/name,raw[key])


if __name__=='__main__':
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--artifacts',type=Path);group.add_argument('--restore-to',type=Path)
    a=p.parse_args()
    if a.restore_to:restore(a.restore_to)
    else:poly(a.artifacts/'3');binary_and_rsm(a.artifacts/'0');gt(a.artifacts/'1');matched(a.artifacts/'2')
