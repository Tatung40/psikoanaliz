"""Collect actual native-R artifacts; audit committed evidence without fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SOURCES=['core/generalizability.py','core/generalizability_advanced.py','scripts/gtheory_reml.R',
 'tests/test_gtheory_advanced.py','validation/gtheory_advanced/reference.R',
 'validation/gtheory_advanced/multivariate_reference.R','validation/gtheory_advanced/PROTOCOL.json']


def sources():
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}


def collect(folder):
    folder=Path(folder);comparisons=sorted(folder.rglob('comparison.json'))
    assert len(comparisons)==11, 'Require ten REML cases plus multivariate comparison'
    records=[{'case':str(p.parent.relative_to(folder)),'result':json.loads(p.read_text())} for p in comparisons]
    assert all(r['result']['status']=='NUMERICALLY VALIDATED' for r in records)
    profile=list(folder.rglob('profile_ci.csv'))
    import pandas as pd
    valid_profile=[p for p in profile if len(pd.read_csv(p))==3 and p.parent.name.startswith('test_r_profile')]
    assert len(valid_profile)==1,'Actual profile/affine/singular test outputs required'
    canonical=[]
    for p in sorted(folder.rglob('*')):
        if p.is_file() and p.suffix in ('.json','.csv','.txt','.log','.stdout','.stderr'):
            canonical.append({'path':str(p.relative_to(folder)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                              'content':p.read_text()})
    return {'status':'NUMERICALLY VALIDATED','scope':'Only executed deterministic fixtures; not blanket certification',
            'source_sha256':sources(),'protocol_commit':'29dd0c2bede3e9cd29114537321bffdb9843aed5',
            'execution_commit':os.getenv('GITHUB_SHA'),'run_id':os.getenv('GITHUB_RUN_ID'),
            'workflow_url':'https://github.com/Tatung40/psikoanaliz/actions/runs/'+os.getenv('GITHUB_RUN_ID',''),
            'comparisons':records,'profile_test':str(valid_profile[0].parent.relative_to(folder)),
            'files':canonical,'facet_status':'NOT EXECUTED: excluded mandatory dependencies',
            'monte_carlo':0}


def audit(path):
    data=json.loads(Path(path).read_text())
    assert data['source_sha256']==sources(),'Reference/campaign sources changed; evidence stale'
    assert len(data['comparisons'])==11 and data['monte_carlo']==0
    assert data['protocol_commit']=='29dd0c2bede3e9cd29114537321bffdb9843aed5'
    assert all(r['result']['status']=='NUMERICALLY VALIDATED' for r in data['comparisons'])
    for f in data['files']:
        assert hashlib.sha256(f['content'].encode()).hexdigest()==f['sha256'],f['path']
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collect');p.add_argument('--output');p.add_argument('--audit');args=p.parse_args()
    if args.audit:
        d=audit(args.audit);print('Archived actual R evidence intact:',len(d['comparisons']),'comparisons; no refits.')
    else:
        d=collect(args.collect);Path(args.output).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
        print('Archived actual R diagnostics:',len(d['comparisons']))
