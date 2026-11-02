"""Reuse actual R execution only when every calibration/source/data byte matches."""
import argparse
import hashlib
import json
import os
from pathlib import Path

REFERENCE=Path('validation/DIF_WALD_LORD_REFERENCE.json')
CASES=('dif1pl_uniform','dif2pl_mixed','dif2pl_none')
SOURCES=('core/irt_dif.py','core/dif_lord.py','validation/dif_wald_lord/compare.py',
         'validation/dif_wald_lord/reference.R','scripts/mirt_irt_b10_reference.R')
FILES=('python_fits.npz','python_lord.csv','python_lord_cov.csv','python_wald.csv','python_wald_cov.csv','settings.json')


def verified_reference():
    evidence=json.loads(REFERENCE.read_text())
    if (evidence['R_execution']!='EXECUTED' or evidence['numerical_status']!='PASS'
            or evidence['chi_relative_limit']!=1e-3 or set(evidence['cases'])!=set(CASES)):
        raise ValueError('Independent R evidence incomplete or failed')
    expected=[str(Path('validation/dif_wald_lord/results',c,f)) for c in CASES for f in FILES]
    for field,paths in [('source_sha256',SOURCES),('python_evidence_sha256',expected)]:
        if set(evidence[field])!=set(paths):raise ValueError('Incomplete reference identity')
        for path in paths:
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=evidence[field][path]:
                raise ValueError('Reference identity changed: '+path)
    for case in CASES:
        record=evidence['cases'][case]
        source=Path('tests/fixtures/irt_b10',f'data_{case}.csv')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=record['R']['data_sha256']:
            raise ValueError('Observed input changed')
        if record['R']['mirt']!='1.48' or not all(record['R']['converged']) or not all(record['R']['bounds_interior']):
            raise ValueError('R convergence/bound gate failed')
        for method in ('lord','wald'):
            metrics=record['comparisons'][method]
            relative=metrics['chi2']['max_relative']
            if metrics['status']!='PASS' or relative is None or relative>1e-3:
                raise ValueError('Independent chi-square criterion failed')
    return evidence


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    try:
        evidence=verified_reference();needs_r=False
    except (ValueError,KeyError,FileNotFoundError) as exc:
        if args.verify:raise
        needs_r=True;print('Actual R fits required:',exc)
    if not needs_r:
        args.output.mkdir(parents=True,exist_ok=True)
        evidence['new_R_fits_this_run']=0
        evidence['reference_reused_with_exact_hashes']=True
        (args.output/'comparison.json').write_text(json.dumps(evidence,indent=2)+'\n')
        print('ACTUAL ARCHIVED R REFERENCE VERIFIED; no new fits; chi-square relative <=1e-3')
    output=os.environ.get('GITHUB_OUTPUT')
    if output:
        with open(output,'a') as f:f.write('needs_r='+str(needs_r).lower()+'\n')
