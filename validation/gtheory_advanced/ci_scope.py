"""Skip unrelated MC only for this closed G-theory expansion scope."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
ALLOWED={'.github/workflows/pytest.yml','.github/workflows/mtk-scientific-gates.yml',
 'core/generalizability_advanced.py','core/academic_result_adapters.py',
 'ui/generalizability_view.py','ui/generalizability_advanced_view.py',
 'scripts/gtheory_reml.R','tests/test_gtheory_advanced.py','tests/test_gtheory_advanced_ui.py',
 'validation/REPRODUCIBLE_ANALYSIS_INVENTORY.json'}


def permitted(paths):
    return bool(paths) and all(p in ALLOWED or p.startswith('validation/gtheory_advanced/') for p in paths)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);args=parser.parse_args()
    paths=subprocess.check_output(['git','diff','--name-only',args.base,'HEAD'],text=True).splitlines()
    if not permitted(paths):raise SystemExit('G-theory-only scope violated; MC suppression rejected')
    for p in ('core/generalizability.py','validation/q1_q3/manifest.json'):
        old=subprocess.check_output(['git','show',args.base+':'+p])
        assert hashlib.sha256(old).digest()==hashlib.sha256((ROOT/p).read_bytes()).digest(),p
    with open(os.environ['GITHUB_OUTPUT'],'a') as stream:stream.write('evidence_only=true\n')
    print('G-theory-only scope and original ANOVA/provenance verified. Zero unrelated MC; all evidence/browser checks remain.')
