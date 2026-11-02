"""One fit per engine/group/density; frozen results remain reusable, never retried."""
import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import subprocess
import zipfile
import numpy as np
import pandas as pd
from core.dif_lord_rasch import lord_1pl_rasch, ROOT
from validation.dif_wald_lord.compare import configuration, discrepancies
from validation.gh_reoptimization.experiment import clean,save

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'


def prepare():
    if (OUT/'python_done.json').exists():raise RuntimeError('Python fits already executed; no repeat')
    OUT.mkdir(exist_ok=True)
    frame,cfg=configuration('dif1pl_uniform','1PL')
    cfg.pop('Lord');cfg.pop('Wald')  # Do not inherit the OLD fixed-SD design labels.
    cfg.update(protocol_commit='9d01c0fdd8681e85e43437906459752353dc003d',
        Python=platform.python_version(),execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        identification='a=1; mean=0; latent SD free in EACH independent group',engine='core.rasch.fit_rasch',
        priors='none',densities=[61,101],native_max_iter=2000,normalized_score_tolerance=1e-6,
        versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','pandas','mirt')})
    save(OUT/'settings.json',cfg)
    for q in (61,101):
        folder=OUT/f'GH{q}';folder.mkdir(exist_ok=True)
        result=lord_1pl_rasch(frame[cfg['items']],frame.Grup,'R','F',anchors=cfg['anchors'],studied=cfg['studied'],points=q)
        result['table'].to_csv(folder/'python_lord.csv',index=False)
        result['covariance'].to_csv(folder/'python_lord_cov.csv',index=False)
        save(folder/'python_diagnostics.json',result['diagnostics'])
        for i,g in enumerate(('R','F')):
            if i>=len(result['fits']):continue
            fit=result['fits'][i];save(folder/f'python_{g}_fit.json',{k:v for k,v in fit.items() if k not in ('data','observed','items','kept_rows')})
            if i>=len(result.get('full_covariances',[])):continue
            pd.DataFrame(dict(item=cfg['items'],b=result['vectors'][i])).to_csv(folder/f'python_{g}_parameters.csv',index=False)
            pd.DataFrame(result['full_covariances'][i]).to_csv(folder/f'python_{g}_full_cov.csv',index=False)
            pd.DataFrame(result['group_covariances'][i]).to_csv(folder/f'python_{g}_b_cov.csv',index=False)
        if len(result.get('vectors',[]))!=2 or not result['table'].p.notna().all():
            raise ValueError('Python Rasch failed; preserved outputs, no repeat/PASS')
    save(OUT/'python_done.json',{'fits_executed':4})


def compare(wide=False):
    from core import rasch
    rroot=OUT/'wide' if wide else OUT
    frame,cfg=configuration('dif1pl_uniform','1PL');rmeta=json.loads((rroot/'r_metadata.json').read_text())
    if rmeta['mirt']!='1.48' or rmeta['data_sha256']!=cfg['data_sha256']:raise ValueError('R/data mismatch')
    report=dict(status='NOT VALIDATED',R_execution='EXECUTED',R_mirt=rmeta['mirt'],R=rmeta,
        covariance='b block after full inverse; latent SD nuisance included',comparisons={},density_stability={},
        limits={'relative_chi_square':1e-3},certification='NO GENERAL DIF-FWER CERTIFICATION')
    passed=True
    for q in (61,101):
        folder=OUT/f'GH{q}';rfolder=rroot/f'GH{q}'
        py=pd.read_csv(folder/'python_lord.csv');rr=pd.read_csv(rfolder/'r_lord.csv')
        if py.Item.tolist()!=rr.Item.tolist() or not (py.df.eq(1).all() and rr.df.eq(1).all()):raise ValueError('Item/df alignment')
        result={key:discrepancies(py[key],rr[key]) for key in ('Statistic','p','Delta_b','SE_b')}
        relative=result['Statistic']['max_relative'];passed &= relative is not None and relative<=1e-3
        result['contrast_covariance']=discrepancies(pd.read_csv(folder/'python_lord_cov.csv').Covariance,pd.read_csv(rfolder/'r_lord_cov.csv').Covariance)
        for g in ('R','F'):
            native=json.loads((folder/f'python_{g}_fit.json').read_text());rdiag=json.loads((rfolder/f'r_{g}_diagnostics.json').read_text())
            raw=pd.read_csv(rfolder/f'r_{g}_raw_parameters.csv');rawV=pd.read_csv(rfolder/f'r_{g}_raw_full_cov.csv',index_col=0)
            variance=float(raw.loc[raw.name.eq('COV_11'),'value'].item())
            # jsonlite defaults round diagnostic scalars; raw CSV values are authoritative.
            r_sd=float(np.sqrt(variance));k=len(cfg['items'])
            assert raw.loc[raw.name.eq('COV_11'),'est'].item() and rawV.shape==(k+1,k+1)
            J=np.zeros((k+1,k+1))
            for j,item in enumerate(cfg['items']):
                par=int(raw.loc[raw.item.eq(item)&raw.name.eq('d'),'parnum'].item())
                J[j,list(rawV.columns).index(f'd.{par}')]=-1
            par=int(raw.loc[raw.name.eq('COV_11'),'parnum'].item())
            J[-1,list(rawV.columns).index(f'COV_11.{par}')]=1/(2*variance)
            r_full=J@rawV.to_numpy()@J.T
            b_py=pd.read_csv(folder/f'python_{g}_parameters.csv').b.to_numpy();b_r=pd.read_csv(rfolder/f'r_{g}_parameters.csv').b.to_numpy()
            data=frame.loc[frame.Grup.eq(g),cfg['items']]
            ll_py=rasch.marginal_loglik(data,b_py,native['sd'],101);ll_r=rasch.marginal_loglik(data,b_r,r_sd,101)
            x,obs=rasch._as_matrix(data);z,lw=rasch._grid(101)
            score=rasch._score(np.r_[b_r,np.log(r_sd)],np.where(obs,x,0.),np.where(obs,1-x,0.),z,lw)/len(x)
            result[g]=dict(b=discrepancies(b_py,b_r),sd_Python=native['sd'],sd_R=r_sd,
                b_covariance=discrepancies(pd.read_csv(folder/f'python_{g}_b_cov.csv'),pd.read_csv(rfolder/f'r_{g}_b_cov.csv')),
                full_b_logSD_covariance=discrepancies(pd.read_csv(folder/f'python_{g}_full_cov.csv'),r_full),
                common_GH101_loglik_Python=ll_py,common_GH101_loglik_R=ll_r,common_loglik_difference=ll_r-ll_py,
                R_solution_common_GH101_score=float(np.max(np.abs(score))),
                Python_converged=native['converged'],R_diagnostics=rdiag)
            passed &= bool(native['converged'] and not native['at_bound'] and rdiag['converged'] and rdiag['interior'] and rdiag['covariance_valid'])
            passed &= bool(np.max(np.abs(score))<=1e-5 and native['max_score_per_person']<=1e-5)
        report['comparisons'][str(q)]=result
    for engine in ('python','r'):
        base=OUT if engine=='python' else rroot
        lo=pd.read_csv(base/'GH61'/f'{engine}_lord.csv');hi=pd.read_csv(base/'GH101'/f'{engine}_lord.csv')
        report['density_stability'][engine]={k:discrepancies(lo[k],hi[k]) for k in ('Statistic','p','Delta_b','SE_b')}
    sources=('core/rasch.py','core/dif_lord.py','core/dif_lord_rasch.py','tests/fixtures/irt_b10/data_dif1pl_uniform.csv')
    report['production_source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    report['comparison_scope']='PASS applies to the preregistered chi-square criterion and convergence/PD/interior gates only; parameter/covariance/common-grid score discrepancies are reported, not independently certified.'
    report['metadata_note']='Effective settings follow actual native fit records and R raw parameter tables; inherited legacy Lord/Wald labels in original settings.json were not used by either fitter. Original workflow report/settings preserved; R SD reconstructed from full-precision raw CSV, not rounded diagnostic JSON.'
    report['status']='NUMERICAL REFERENCE PASS / NOT FWER CERTIFIED' if passed else 'NUMERICAL REFERENCE FAIL'
    report['R_window']='[-12,12]' if wide else 'default[-6,6]'
    if wide:save(rroot/'report.json',report)
    save(OUT/'report.json',report)
    if not passed:raise ValueError('Independent numerical comparison FAIL; evidence preserved')


def emit():
    """Lossless log transport of compact CSV/JSON evidence; original artifact also uploaded."""
    files=[p for p in OUT.rglob('*') if p.suffix in ('.json','.csv','.log','.txt')]
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.writestr(str(p.relative_to(OUT)),p.read_bytes())
    content=archive.getvalue();print('RASCH_EVIDENCE_SHA256='+hashlib.sha256(content).hexdigest())
    print('RASCH_EVIDENCE_BASE64='+base64.b64encode(content).decode())

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--compare',action='store_true');parser.add_argument('--emit',action='store_true');parser.add_argument('--wide',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    if args.compare:compare(args.wide)
    if args.emit:emit()
