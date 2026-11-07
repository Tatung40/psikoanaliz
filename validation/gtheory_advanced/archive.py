"""Archive and independently recheck saved actual-R outputs. Never refit."""
import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
from validation.gtheory_advanced.verify import saved_case

ROOT=Path(__file__).resolve().parents[2]
SOURCES=['core/generalizability.py','core/generalizability_advanced.py','scripts/gtheory_reml.R',
 'tests/test_gtheory_advanced.py','validation/gtheory_advanced/reference.R',
 'validation/gtheory_advanced/multivariate_reference.R','validation/gtheory_advanced/PROTOCOL.json']
AUDIT=['validation/gtheory_advanced/verify.py','validation/gtheory_advanced/archive.py']


def _digests(paths):return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}


def _records(files):
    comparisons=[]
    cases=sorted({str(Path(f['path']).parent) for f in files if Path(f['path']).name=='settings.json'})
    for case in cases:
        selected={f['path'][len(case)+1:]:f['content'] for f in files if f['path'].startswith(case+'/')}
        if 'python_variance.csv' in selected:
            comparisons.append({'case':case,'result':saved_case(selected)})
    assert len(comparisons)==10,'Require all ten actual REML cases, including boundary failures'
    multivariate=[f for f in files if f['path'].endswith('/comparison.json') and 'matrix_max_differences' in json.loads(f['content'])]
    assert len(multivariate)==1
    comparisons.append({'case':str(Path(multivariate[0]['path']).parent),'result':json.loads(multivariate[0]['content'])})
    return comparisons


def collect(folder):
    folder=Path(folder);files=[]
    for p in sorted(folder.rglob('*')):
        relative=p.relative_to(folder)
        if any(part.endswith('current') for part in relative.parts):continue
        if p.is_file() and p.suffix in ('.json','.csv','.txt','.log','.stdout','.stderr'):
            files.append({'path':str(relative),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'content':p.read_text()})
    records=_records(files)
    profile=[f for f in files if Path(f['path']).name=='profile_ci.csv' and Path(f['path']).parent.name.startswith('test_r_profile')]
    assert len(profile)==1
    import pandas as pd
    intervals=pd.read_csv(io.StringIO(profile[0]['content']))
    assert len(intervals)==3 and (intervals.lower>=0).all() and (intervals.upper>=intervals.lower).all()
    return {'status':'PARTIAL / NUMERICALLY VALIDATED OUTPUTS AND CONFIRMED SAFETY REJECTIONS',
            'scope':'Only executed deterministic fixtures; singular fits remain NOT CERTIFIED',
            'source_sha256':_digests(SOURCES),'audit_sha256':_digests(AUDIT),
            'execution_sources_sha256':_digests(SOURCES),
            'protocol_commit':'29dd0c2bede3e9cd29114537321bffdb9843aed5',
            'execution_commit':os.getenv('GITHUB_SHA'),'run_id':os.getenv('GITHUB_RUN_ID'),
            'workflow_url':'https://github.com/Tatung40/psikoanaliz/actions/runs/'+os.getenv('GITHUB_RUN_ID',''),
            'comparisons':records,'profile_test':str(Path(profile[0]['path']).parent),
            'profile_intervals':intervals.to_dict('records'),'files':files,
            'facet_status':'NOT EXECUTED: excluded mandatory dependencies','monte_carlo':0}


def audit(path):
    data=json.loads(Path(path).read_text())
    assert data['source_sha256']==_digests(SOURCES),'Sources changed; evidence stale'
    assert data['audit_sha256']==_digests(AUDIT),'Audit implementation changed'
    assert data['monte_carlo']==0 and data['protocol_commit']=='29dd0c2bede3e9cd29114537321bffdb9843aed5'
    for p,h in data['execution_sources_sha256'].items():
        if p!='tests/test_gtheory_advanced.py':assert h==data['source_sha256'][p],p
    old=data.get('executed_test_source')
    if old:
        assert hashlib.sha256(old.encode()).hexdigest()==data['execution_sources_sha256']['tests/test_gtheory_advanced.py']
        def fixture(source):return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='fixture'))
        assert fixture(old)==fixture((ROOT/'tests/test_gtheory_advanced.py').read_text()),'DGP changed after execution'
    for f in data['files']:
        assert hashlib.sha256(f['content'].encode()).hexdigest()==f['sha256'],f['path']
    assert _records(data['files'])==data['comparisons'],'Saved numerical comparisons disagree'
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collect');p.add_argument('--output');p.add_argument('--audit');args=p.parse_args()
    if args.audit:
        d=audit(args.audit);print('Actual R archive verified without refitting:',len(d['comparisons']),'comparisons.')
    else:
        d=collect(args.collect);Path(args.output).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
        print('Archived actual R outputs:',len(d['comparisons']))
