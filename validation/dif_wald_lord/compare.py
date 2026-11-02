"""Three fixed B10 fixtures; cached actual fits, no MC or automatic refit/retry."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import mirt
from mirt.estimation.quadrature import GaussHermiteQuadrature
from core import irt_dif as W
from core.dif_lord import lord_dif,lord_from_parameters

FIXTURES=Path('tests/fixtures/irt_b10')
CASES={'dif1pl_uniform':'1PL','dif2pl_mixed':'2PL','dif2pl_none':'2PL'}
CHI_RELATIVE_LIMIT=1e-3  # User-predeclared; never calibrated from observed results.


def configuration(case,model):
    path=FIXTURES/f'data_{case}.csv';frame=pd.read_csv(path);items=list(frame.columns[1:])
    original_n=len(frame);frame=frame.dropna(subset=items).copy()
    anchors=(pd.read_csv(FIXTURES/f'py_dif_anchors_{case}.csv').anchor.tolist() if case!='dif2pl_none' else items[:8])
    return frame,dict(case=case,model=model,items=items,anchors=anchors,studied=[i for i in items if i not in anchors],
        original_n=original_n,used_n=len(frame),excluded_incomplete=original_n-len(frame),
        missing_policy='Explicit identical complete-case cohort for both methods; no production imputation',
        data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),mirt=mirt.__version__,Python=platform.python_version(),
        Lord=dict(n_quadpts=61,tol=1e-8,max_iter=3000,estimation='ML',prior='none',latent='N(0,1)'),
        Wald=dict(quadrature='mirt',points=61,latent='reference N(0,1), focal mean/SD free',priors='none'),
        covariance_condition_limit=1e8,chi_relative_limit=CHI_RELATIVE_LIMIT)


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    q=GaussHermiteQuadrature(n_points=61,n_dimensions=1)
    pd.DataFrame({'theta':q.nodes[:,0],'weight':q.weights}).to_csv(root/'integration.csv',index=False)
    for case,model in CASES.items():
        frame,cfg=configuration(case,model);out=root/case;out.mkdir(exist_ok=True)
        arrays=out/'python_fits.npz';meta=out/'settings.json'
        if arrays.exists():
            previous=json.loads(meta.read_text())
            for key in ('model','items','anchors','studied','data_sha256','mirt','Lord','Wald'):
                if previous[key]!=cfg[key]:raise ValueError('Cached fit settings/data changed; no silent refit')
            continue
        w=W.wald_dif(frame[cfg['items']],frame.Grup,'R','F',model=model,anchors=cfg['anchors'],
                     studied=cfg['studied'],quadrature='mirt',points=61)
        l=lord_dif(frame[cfg['items']],frame.Grup,'R','F',model=model,anchors=cfg['anchors'],studied=cfg['studied'])
        # Preserve actual fits even when invalid; derived comparison must fail visibly.
        wf=w['fit'];fits=l['fits']
        cfg['native_diagnostics']={key:wf[key] for key in ('converged','loglik','iterations','max_score_per_person','mu_F','sd_F')}
        cfg['Lord_diagnostics']=[dict(converged=bool(f.converged),iterations=int(f.n_iterations),loglik=float(f.log_likelihood),
            original_vcov_labels=list(f.vcov_labels() if callable(f.vcov_labels) else f.vcov_labels)) for f in fits]
        cfg['native_labels']=wf['design'].labels
        cfg['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        meta.write_text(json.dumps(cfg,indent=2,default=lambda x:x.item() if hasattr(x,'item') else x)+'\n')
        w['table'].to_csv(out/'python_wald.csv',index=False);w['covariance'].to_csv(out/'python_wald_cov.csv',index=False)
        l['table'].to_csv(out/'python_lord.csv',index=False);l['covariance'].to_csv(out/'python_lord_cov.csv',index=False)
        if 'vectors' not in l:raise ValueError('Lord invalid; raw diagnostics preserved, no success claim')
        np.savez_compressed(arrays,w_params=wf['params'],w_cov=wf['cov'],w_bound=wf['at_bound'],
            r=l['vectors'][0],f=l['vectors'][1],Vr=l['group_covariances'][0],Vf=l['group_covariances'][1])
        if not w['table'].p.notna().all() or not l['table'].p.notna().all():
            raise ValueError('Invalid covariance/fit; no retry or PASS')


def discrepancies(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Incomparable shape/nonfinite')
    diff=np.abs(a-b);den=np.abs(b)
    relative=np.divide(diff,den,out=np.where(diff==0,0.,np.inf),where=den>0)
    maximum=float(relative.max(initial=0))
    return dict(max_absolute=float(diff.max(initial=0)),max_relative=maximum if np.isfinite(maximum) else None,
                undefined_relative_count=int(np.sum((den==0)&(diff>0))))


def compare(root):
    report={'status':'MEASURED / NOT CERTIFIED','R_execution':'EXECUTED','cases':{}}
    failed=False
    for case,model in CASES.items():
        out=root/case;cfg=json.loads((out/'settings.json').read_text())
        rmeta=json.loads((out/'r_settings.json').read_text())
        if rmeta['data_sha256']!=cfg['data_sha256'] or str(rmeta['mirt'])!='1.48':raise ValueError('R input/version mismatch')
        record={'R':rmeta,'comparisons':{}};archive=np.load(out/'python_fits.npz')
        for method in ('wald','lord'):
            py=pd.read_csv(out/f'python_{method}.csv').set_index('Item').loc[cfg['studied']]
            rr=pd.read_csv(out/f'r_{method}.csv').set_index('Item').loc[cfg['studied']]
            pc=pd.read_csv(out/f'python_{method}_cov.csv').sort_values(['Item','Row','Column'])
            rc=pd.read_csv(out/f'r_{method}_cov.csv').sort_values(['Item','Row','Column'])
            stat=discrepancies(py.Statistic,rr.Statistic)
            cols=['Delta_b'] if model=='1PL' else ['Delta_a','Delta_b']
            metrics=dict(chi2=stat,delta=discrepancies(py[cols],rr[cols]),
                covariance=discrepancies(pc.Covariance,rc.Covariance))
            metrics['status']='PASS' if stat['max_relative'] is not None and stat['max_relative']<=CHI_RELATIVE_LIMIT else 'FAIL'
            # Both convergence, physical bounds, covariance gates must also pass.
            if not all(rmeta['converged']) or not all(rmeta['bounds_interior']):metrics['status']='FAIL'
            if not cfg['native_diagnostics']['converged'] or not all(d['converged'] for d in cfg['Lord_diagnostics']):metrics['status']='FAIL'
            failed |= metrics['status']!='PASS';record['comparisons'][method]=metrics
        for group,key in [('R','r'),('F','f')]:
            rv=pd.read_csv(out/f'r_lord_{group}_parameters.csv').value.to_numpy()
            rc=pd.read_csv(out/f'r_lord_{group}_cov.csv').to_numpy()
            record['comparisons']['single_'+group]=dict(parameters=discrepancies(archive[key],rv),
                covariance=discrepancies(archive['V'+key],rc))
        report['cases'][case]=record
    report['numerical_status']='FAIL' if failed else 'PASS'
    (root/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if failed:raise SystemExit('Independent R comparison FAIL; original outputs preserved')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('validation/dif_wald_lord/results'))
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--compare',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare(args.output)
    if args.compare:print(json.dumps(compare(args.output),indent=2))
