"""GRM/PCM fixed-parameter checks only. Never calibrates or simulates responses."""
import argparse
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types
import numpy as np
import pandas as pd
from scipy.special import roots_hermite

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
HARNESS_COMMIT='18d1bcf72c85a174ad278ae4d8ea925b412c0275'


@lru_cache(maxsize=1)
def helpers():
    # Reuse the exact PR63 helpers without copying them or modifying that PR.
    r=subprocess.run(['git','show',HARNESS_COMMIT+':validation/r_mirt_phase1/harness.py'],
        cwd=ROOT,capture_output=True,text=True,check=True)
    module=types.ModuleType('pr63_preflight_helpers')
    module.__file__=str(ROOT/'validation/r_mirt_phase1/harness.py')
    exec(compile(r.stdout,module.__file__,'exec'),module.__dict__)
    return module,hashlib.sha256(r.stdout.encode()).hexdigest()


def probes(data, output):
    import mirt
    from core.irt_b14 import prepare_polytomous
    h,_=helpers();x,labels=prepare_polytomous(data,'GRM')
    C=len(labels);k=len(x.columns)
    if C!=4 or any(set(x[col])!=set(range(C)) for col in x):
        raise ValueError('Expected the existing complete four-category fixture')
    nodes,weights=roots_hermite(61);nodes*=np.sqrt(2);weights/=np.sqrt(np.pi)
    pd.DataFrame(dict(theta=nodes,weight=weights)).to_csv(output/'integration.csv',index=False)
    theta=np.unique(np.r_[nodes,np.linspace(-4,4,81),-1.,1.])
    pd.DataFrame(dict(node=np.arange(len(theta)),theta=theta)).to_csv(output/'evaluation_grid.csv',index=False)
    curves={};diagnostics={}
    for name in ('GRM','PCM'):
        b=np.linspace(-1,1,k)[:,None]+np.array([-1.,0.,1.])
        a=np.linspace(.8,1.8,k) if name=='GRM' else np.ones(k)
        if name=='PCM': b[0]=[.5,-.5,.2]  # Unordered steps are legitimate, never sort.
        model=(mirt.GradedResponseModel if name=='GRM' else mirt.PartialCreditModel)(k,C)
        if name=='GRM':model.set_parameters(discrimination=a,thresholds=b)
        else:model.set_parameters(steps=b)
        pd.DataFrame(dict(item=x.columns,a=a,b1=b[:,0],b2=b[:,1],b3=b[:,2])).to_csv(output/f'{name}_physical.csv',index=False)
        P=model.probability(theta);I=np.column_stack([model.information(theta,item_idx=i) for i in range(k)])
        frame=pd.DataFrame([dict(item=str(item),node=q,theta=t,category=c,probability=float(P[q,i,c]),information=float(I[q,i]))
            for i,item in enumerate(x.columns) for q,t in enumerate(theta) for c in range(C)])
        frame.to_csv(output/f'{name}_python_curves.csv',index=False);curves[name]=frame
        parameters=[h.semantic_parameters(model.probability(np.array([-1.,1.]))[:,i],np.array([-1.,1.]),name) for i in range(k)]
        mapping=max(max(abs(np.asarray(p['b'])-b[i]).max(),abs(np.asarray(p['a'])-a[i]).max()) for i,p in enumerate(parameters))
        if mapping>np.sqrt(np.finfo(float).eps):raise ValueError('Python physical mapping mismatch')
        if not np.allclose(P.sum(2),1) or not np.isfinite(I).all():raise ValueError('Invalid fixed-parameter Python probe')
        diagnostics[name]=dict(mapping_max_absolute=float(mapping),probability_sum_error=float(abs(P.sum(2)-1).max()),
            category_codes=list(range(C)),estimation='NONE',n_parameters_probed=k*C)
    return curves,diagnostics


def run(output, rscript=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    h,digest=helpers();source=ROOT/h.SOURCES['GRM'];data=pd.read_csv(source)
    curves,checks=probes(data,output)
    data.to_csv(output/'data.csv',index=False)
    result=dict(status='NOT EXECUTED',R_execution='NOT EXECUTED',scientific_pass=False,fits=0,simulations=0,
        seed=20261024,fixture=str(source.relative_to(ROOT)),fixture_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        reused_harness_commit=HARNESS_COMMIT,reused_harness_sha256=digest,
        python_versions={k:importlib.metadata.version(k) for k in ('mirt','numpy','scipy','pandas')},
        python_checks=checks,R_checks='PENDING',
        script_sha256={name:hashlib.sha256((ROOT/'validation'/name).read_bytes()).hexdigest() for name in ('poly_r_preflight.py','poly_r_preflight.R')},
        algebra_roundoff_rtol=float(np.sqrt(np.finfo(float).eps)),
        notice='Fixed-probe numerical equality uses R all.equal default sqrt(eps), NOT a scientific fit acceptance tolerance. No scientific PASS without independent fits.',
        GH=dict(points=61,distribution='N(0,1)',estimator_integration='NOT EXECUTED'))
    executable=rscript or shutil.which('Rscript')
    if executable:
        try:
            r=subprocess.run([executable,str(ROOT/'validation/poly_r_preflight.R'),str(output)],capture_output=True,text=True,timeout=60)
            result['R_execution']='EXECUTED'
            (output/'R_stdout.log').write_text(r.stdout);(output/'R_stderr.log').write_text(r.stderr)
            if r.returncode:raise ValueError(f'R preflight exit {r.returncode}; see stderr')
            meta=pd.read_csv(output/'R_metadata.csv').iloc[0].to_dict()
            if str(meta['mirt_version'])!='1.48':raise ValueError('Wrong R mirt version')
            result['R_metadata']=meta
            rc=pd.read_csv(output/'R_checks.csv')
            if rc.empty or not rc.checked.eq(True).all():raise ValueError('Incomplete/failed R checks')
            result['R_checks']=rc.to_dict('records')
            result['comparisons']={}
            for model,py in curves.items():
                ref=pd.read_csv(output/f'{model}_R_curves.csv')
                metrics=h.compare_curves(py,ref);result['comparisons'][model]=metrics
                keys=['item','node','category']
                pair=py.merge(ref,on=keys,suffixes=('_python','_R'),validate='one_to_one')
                if len(pair)!=len(py) or len(ref)!=len(py):raise ValueError('Missing category/grid keys')
                for col in ('theta','probability','information'):
                    if not np.allclose(pair[col+'_python'],pair[col+'_R'],rtol=result['algebra_roundoff_rtol'],atol=result['algebra_roundoff_rtol']):
                        raise ValueError(model+' fixed-probe '+col+' mismatch')
            result['status']='PREFLIGHT CHECKED / NOT SCIENTIFICALLY VALIDATED'
        except (OSError,ValueError,subprocess.TimeoutExpired) as error:
            if isinstance(error,subprocess.TimeoutExpired):
                result['R_execution']='EXECUTED'
                for key,value in [('R_stdout.log',error.stdout),('R_stderr.log',error.stderr)]:
                    (output/key).write_text(value.decode(errors='replace') if isinstance(value,bytes) else value or '')
            result.update(status='FAIL',reason=str(error))
    (output/'report.json').write_text(json.dumps(result,indent=2,default=str,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=run(args.output);print(result['status'])
    if result['status']=='FAIL':raise SystemExit(1)
