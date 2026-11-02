"""stdlib-only CI cache gate: immutable evidence, never manufacture R execution."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'validation/lord_rasch_free_scale/results'


def verify():
    report=OUT/'wide/report.json'
    if not report.exists():
        needs=True
    else:
        evidence=json.loads(report.read_text())
        if evidence['status']!='NUMERICAL REFERENCE PASS / NOT FWER CERTIFIED' or evidence['R_execution']!='EXECUTED':
            raise ValueError('Existing scientific failure; no automatic refit')
        for p,h in evidence['production_source_sha256'].items():
            if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('Reference source/data identity mismatch')
        for p,h in json.loads((OUT/'SHA256.json').read_text()).items():
            if hashlib.sha256((OUT/p).read_bytes()).hexdigest()!=h:raise ValueError('Raw reference evidence changed')
        needs=False
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'],'a') as stream:
            stream.write(f'needs_r={str(needs).lower()}\nneeds_python={str(not (OUT/"python_done.json").exists()).lower()}\n')
    print('PENDING ACTUAL R EXECUTION' if needs else 'Actual independent R evidence verified; no new fits')

if __name__=='__main__':verify()
