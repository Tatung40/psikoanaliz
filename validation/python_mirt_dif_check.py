"""Small reproducible null diagnostic; no certification or local reference engine.

PYTHONPATH=. python validation/python_mirt_dif_check.py --output /tmp/dif-evidence
Known-fixture snapshot is captured by PSIKO_DIF_EVIDENCE during pytest.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import mirt
from scipy.stats import beta
from core.python_mirt_dif import analyze


def save_snapshot(result,path):
    """Keep full fit settings, parameters, signed LR and missing inference explicitly."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(dict(metadata=result['metadata'],tables={k:json.loads(v.to_json(orient='records',double_precision=15))
                    for k,v in result['tables'].items() if k not in ('Curves','Difference','IIF','TIF')}),ensure_ascii=False,indent=2,allow_nan=False)
    path.write_text(text+'\n',encoding='utf-8')


def null_diagnostic(repetitions=12):
    """Predeclared 12 seeds, N=600, J=6, one item, explicit other-item anchors.

    Diagnostic chi-square reference only; not the production decision gate.
    Twelve runs cannot establish size/FDR control. Retain failed runs separately.
    """
    from mirt.multigroup.invariance import invariance_lrt
    rows=[]
    for seed in range(20261026,20261026+repetitions):
        # One fixed-parameter library-generated sample randomly partitioned; no DIF by design.
        x=mirt.simdata(model='2PL',n_persons=600,n_items=6,seed=seed)
        rng=np.random.default_rng(seed);g=np.repeat(['R','F'],300);rng.shuffle(g)
        d=pd.DataFrame(x,columns=list('ABCDEF'));d['G']=g
        result=analyze(d,items=list('ABCDEF'),group='G',reference='R',focal='F',model='2PL',
                       anchors=list('ABCDE'),studied=['F'],n_quadpts=41,tol=1e-5)
        H0=result['fits']['H0:F'];H1=result['fits']['H1'];lr=result['tables']['DIF'].LR.item()
        valid=bool(H0.converged and H1.converged and lr>=0)
        p=invariance_lrt(H0,H1)['p_value'] if valid else None
        rows.append(dict(seed=seed,converged=bool(H0.converged and H1.converged),LR=lr,
                         df=H1.n_parameters-H0.n_parameters,diagnostic_chi2_p=p,
                         H0_logLik=H0.log_likelihood,H1_logLik=H1.log_likelihood,
                         N=600,data_sha256=result['metadata']['Data_SHA256']))
    valid=[r for r in rows if r['diagnostic_chi2_p'] is not None];n=len(valid)
    k=sum(r['diagnostic_chi2_p']<.05 for r in valid)
    interval=[float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.] if n else None
    return dict(status='NOT CERTIFIED',R_execution='NOT EXECUTED',engine='Python mirt '+mirt.__version__,
        design='2PL intercept-only, explicit fixed anchors; 12-seed diagnostic, not FDR validation',
        alpha=.05,repetitions=repetitions,valid=n,failures=repetitions-n,diagnostic_rejections=k,
        diagnostic_rejection_rate=k/n if n else None,exact_binomial_CI95=interval,rows=rows,
        limitation='Not independent R validation; small conditional-size diagnostic; cannot certify Type-I/FDR, other models or data-driven anchors. Production p/q remain disabled.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    result=null_diagnostic();(args.output/'null.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print({k:v for k,v in result.items() if k!='rows'})
