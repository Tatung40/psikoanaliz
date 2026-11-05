"""Fail-closed evidence-only CI selection: never suppress studies for engine changes."""
import argparse
import os
from pathlib import PurePosixPath
import subprocess

EXACT={'.github/workflows/mtk-scientific-gates.yml','.github/workflows/pytest.yml',
       'tests/test_certification_q123.py','validation/FOUR_PL_ML_DECISION.md',
       'validation/CERTIFICATION_PREAUDIT.md','validation/CODEX_CERTIFICATION_QUEUE.md',
       'validation/SCIENTIFIC_MODEL_CERTIFICATION_MATRIX.md'}


def permitted(paths):
    return bool(paths) and all(p in EXACT or p.startswith('validation/q1_q3/') or
                              (PurePosixPath(p).parent==PurePosixPath('validation/evidence') and
                               PurePosixPath(p).name in ('r_polytomous.json','r_rsm.json','r_34.json','r_gt.json'))
                              for p in paths)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True)
    args=parser.parse_args()
    paths=subprocess.check_output(['git','diff','--name-only',args.base,'HEAD'],text=True).splitlines()
    if not permitted(paths):raise SystemExit('Evidence-only marker rejected: changes outside audited Q1-Q3 scope')
    with open(os.environ['GITHUB_OUTPUT'],'a') as stream:stream.write('evidence_only=true\n')
    print('ZERO MONTE CARLO: immutable evidence tests and all browser checks remain mandatory.')
