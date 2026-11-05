"""Run the Java fit once (CI), then apply the PR #146 common-evaluator gates."""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from validation.q1_q3.profile import ROOT, gh
from validation.r_mirt_phase1.equivalence_audit import BOUNDS
from validation.four_pl_engine.compare import DATA, curves, gates

HERE = Path(__file__).resolve().parent


def main(output, classpath):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    z, w = gh(61); pd.DataFrame({'node': z, 'weight': w}).to_csv(output / 'gh61.csv', index=False)
    run = subprocess.run(['java', '-cp', classpath, 'Fit4PL', str(DATA), str(output / 'gh61.csv'), str(output / 'java.csv')],
                         capture_output=True, text=True)
    rep = dict(status='FIXTURE NUMERICAL COMPARISON / NOT CERTIFIED', monte_carlo_executions=0, exit=run.returncode,
               stderr_tail=run.stderr[-3000:])
    if run.returncode == 0:
        x = pd.read_csv(DATA); X = x.to_numpy()
        p = pd.read_csv(output / 'java.csv').set_index('item').loc[x.columns, ['a', 'b', 'c', 'd']]
        log = (output / 'java.csv.log').read_text(); rep['em_log_tail'] = log[-1500:]
        outside = [f'{i}:{k}' for i in p.index for k in p.columns
                   if not (BOUNDS[k][0] - 1e-9 <= p.loc[i, k] <= BOUNDS[k][1] + 1e-9) or not np.isfinite(p.loc[i, k])]
        rep.update(parameters=p.to_dict('index'), outside_common_bounds=outside)
        if outside:
            rep['PASS'] = False; rep['blocker'] = 'estimates outside the common parameter space (unconstrained optimizer)'
        else:
            rep.update(gates(X, p, 61))
            py = json.loads((ROOT / 'validation/four_pl_gh_refinement/result.json').read_text())
            pp = pd.DataFrame.from_dict(py['parameters'], orient='index').loc[x.columns, ['a', 'b', 'c', 'd']]
            P, I = curves(p, np.linspace(-4, 4, 161)); Pp, Ip = curves(pp, np.linspace(-4, 4, 161))
            rep['vs_python'] = dict(max_param=float((p - pp).abs().to_numpy().max()), max_ICC=float(np.abs(P - Pp).max()),
                                    max_IIF=float(np.abs(I - Ip).max()))
    else:
        rep['PASS'] = False
    (output / 'java_report.json').write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep.get(k) for k in ('exit', 'PASS', 'blocker', 'outside_common_bounds', 'score_objective', 'score_GH161',
                                               'logLik', 'gate', 'boundaries', 'vs_python', 'parameters')}, indent=1, default=float))
    print('STDERR', rep['stderr_tail'][-1500:]); print('EMLOG', rep.get('em_log_tail', '')[-800:])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--output', required=True); ap.add_argument('--classpath', required=True)
    main(**vars(ap.parse_args()))
