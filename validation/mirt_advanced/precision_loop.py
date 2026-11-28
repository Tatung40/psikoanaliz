"""Advanced MIRT closed loop, iteration 1 (precision hypothesis).

Hypothesis (predeclared, see LEDGER.md): GRM_EFA_2D and Bifactor_4D miss the
probability tolerance because the Python EM stopped before stationarity.
Smallest diagnostic: one fresh independent R + Python calibration with the
UNCHANGED stage-1 criteria, then the UNCHANGED stage-1 Python precision
continuation (tol 1e-11). No criteria, data, grid or tolerance changes.
The shared stage-1 harness is imported read-only; nothing in it is modified.
"""
import argparse, hashlib, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mirt_stage1'))
import compare  # noqa: E402  (read-only reuse of the shared harness)

PROFILES = ('GRM_EFA_2D', 'Bifactor_4D')
CRITERIA_SHA = hashlib.sha256((compare.ROOT / 'criteria.json').read_bytes()).hexdigest()

def main(out):
    out.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get('GITHUB_RUN_ID', 'local')
    summary = dict(iteration=1, criteria_sha256=CRITERIA_SHA, run_id=run_id, profiles={})
    for name in PROFILES:
        base = compare.run(name, out / 'baseline' / name)
        rec = dict(baseline_status=base['status'], baseline_blocker=base.get('blocker'))
        if base['status'] == 'NUMERICALLY VALIDATED / NOT CERTIFIED':
            rec['decision'] = 'PASS (baseline)'
        elif not (base.get('R', {}).get('converged') and base.get('Python', {}).get('converged')):
            rec['decision'] = 'BLOCKED: precision continuation requires converged baseline fits'
        else:
            try:
                prec = compare.precision(name, out / 'baseline', out / 'precision' / name)
                prec['settings']['source_run'] = run_id  # actual provenance of the retained R fit
                (out / 'precision' / name / 'report.json').write_text(json.dumps(prec, indent=2, allow_nan=False) + '\n')
                rec['precision_status'] = prec['status']; rec['precision_blocker'] = prec.get('blocker')
                rec['failed_checks'] = sorted(k for k, v in prec.get('comparison', {}).items() if v['status'] != 'PASS')
                rec['decision'] = 'PASS' if prec['status'] == 'NUMERICALLY VALIDATED / NOT CERTIFIED' else 'FAIL'
            except Exception as e:  # recorded, never concealed
                rec['decision'] = 'BLOCKED'; rec['precision_blocker'] = str(e)
        rec['base_failed_checks'] = sorted(k for k, v in base.get('comparison', {}).items() if v['status'] != 'PASS')
        summary['profiles'][name] = rec
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
    print(json.dumps(summary, indent=2, default=str))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    main(p.parse_args().output)
