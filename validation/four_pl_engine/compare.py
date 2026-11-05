"""Execute PROTOCOL.json: new R mirt fits (GH61, GH101) + common-evaluator gates; reuse PR #145 Python result."""
import argparse, hashlib, json, subprocess, time
from pathlib import Path

import numpy as np
import pandas as pd

from validation.q1_q3.profile import ROOT, gh
from validation.r_mirt_phase1.equivalence_audit import BOUNDS, evaluate
from validation.r_mirt_phase1.information_audit import information
from validation.r_mirt_phase1.m12_diagnostic import feasible_scores, scores

HERE = Path(__file__).resolve().parent
DATA = ROOT / 'tests/fixtures/irt_b11/data_irt4pl_ok.csv'


def curves(p, theta):
    a, b, c, d = (p[k].to_numpy() for k in ('a', 'b', 'c', 'd'))
    e = 1 / (1 + np.exp(-a * (theta[:, None] - b))); P = c + (d - c) * e
    return P, (a * (d - c) * e * (1 - e)) ** 2 / (P * (1 - P))


def gates(X, p, objective_q):
    out = {'logLik': {f'GH{n}': evaluate(X, p, *gh(n))[2] for n in (61, 101, 161)}}
    out['score_objective'] = float(np.abs(feasible_scores(p, scores(X, p, *gh(objective_q))).to_numpy()).max())
    out['score_GH161'] = float(np.abs(feasible_scores(p, scores(X, p, *gh(161))).to_numpy()).max())
    z, w = gh(objective_q); sc = lambda v: scores(X, v, z, w)
    g = sc(p).to_numpy().ravel(); v = p.to_numpy().ravel(); lo = [b[0] for b in BOUNDS.values()]; hi = [b[1] for b in BOUNDS.values()]
    free = [j for j, q in enumerate(v) if not ((q <= lo[j % 4] + 1e-10 and g[j] <= 0) or (q >= hi[j % 4] - 1e-10 and g[j] >= 0))]
    h, _ = information(sc, p, float(np.cbrt(np.finfo(float).eps))); face = ((h + h.T) / 2)[np.ix_(free, free)]
    eig = np.linalg.eigvalsh(face)
    out.update(information_min=float(eig[0]), condition=float(np.linalg.cond(face)),
               boundaries=[f'{i}:{k}' for i in p.index for k in p.columns
                           if abs(p.loc[i, k] - BOUNDS[k][0]) <= 1e-6 or abs(p.loc[i, k] - BOUNDS[k][1]) <= 1e-6])
    obj = out['logLik'][f'GH{objective_q}']
    out['gate'] = dict(stationary_objective=out['score_objective'] <= 1e-5,
                       audit_GH161=out['score_GH161'] <= 1e-5 and abs(out['logLik']['GH161'] - obj) <= 1e-3,
                       information=out['information_min'] > 0 and out['condition'] <= 1e8)
    out['PASS'] = all(out['gate'].values())
    return out


def main(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((HERE / 'PROTOCOL.json').read_text())
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == protocol['data']['sha256']
    x = pd.read_csv(DATA); X = x.to_numpy()
    report = dict(status='FIXTURE NUMERICAL COMPARISON / NOT CERTIFIED', monte_carlo_executions=0, engines={})
    alt = output / 'r_alternatives.json'
    subprocess.run(['Rscript', str(HERE / 'r_alternatives.R'), str(alt)], check=False)
    report['R_alternatives'] = json.loads(alt.read_text()) if alt.exists() else 'NOT EXECUTED'
    solutions = {}
    for q in (61, 101):
        d = output / f'R_GH{q}'; t = time.monotonic()
        proc = subprocess.run(['Rscript', str(HERE / 'r_fit.R'), str(DATA), str(q), str(d)], capture_output=True, text=True)
        rec = dict(exit=proc.returncode, stderr_tail=proc.stderr[-2000:], wall_seconds=time.monotonic() - t)
        if proc.returncode == 0:
            p = pd.read_csv(d / 'parameters.csv').set_index('item').loc[x.columns, ['a', 'b', 'c', 'd']]
            meta = pd.read_csv(d / 'metadata.csv').iloc[0].to_dict()
            inside = all(((p[k] >= lo - 1e-9) & (p[k] <= hi + 1e-9)).all() for k, (lo, hi) in BOUNDS.items())
            rec.update(metadata=meta, inside_bounds=bool(inside), parameters=p.to_dict('index'))
            if inside and np.isfinite(p.to_numpy()).all():
                rec.update(gates(X, p, q)); solutions[f'R_GH{q}'] = p
            else:
                rec['PASS'] = False
        else:
            rec['PASS'] = False
        report['engines'][f'R_mirt_GH{q}'] = rec
    py = json.loads((ROOT / 'validation/four_pl_gh_refinement/result.json').read_text())
    pp = pd.DataFrame.from_dict(py['parameters'], orient='index').loc[x.columns, ['a', 'b', 'c', 'd']]
    report['engines']['python_mirt_GH101_PR145'] = dict(reused=True, **gates(X, pp, 101))
    theta = np.linspace(-4, 4, 161); Pp, Ip = curves(pp, theta)
    for name, p in solutions.items():
        P, I = curves(p, theta)
        report['engines'][f'R_mirt_{name[2:]}']['vs_python'] = dict(
            logLik_GH161=report['engines'][f'R_mirt_{name[2:]}']['logLik']['GH161'] - report['engines']['python_mirt_GH101_PR145']['logLik']['GH161'],
            max_param=float((p - pp).abs().to_numpy().max()), max_ICC=float(np.abs(P - Pp).max()), max_IIF=float(np.abs(I - Ip).max()))
    passed = [k for k, v in report['engines'].items() if v.get('PASS')]
    alt_ok = isinstance(report['R_alternatives'], dict) and any(
        v.get('upper_asymptote_argument') or v.get('four_pl_functions') for v in report['R_alternatives'].values() if isinstance(v, dict))
    report['decision'] = ('A' if any(k.startswith('R_mirt') for k in passed) else
                          'C' if 'python_mirt_GH101_PR145' in passed else 'D')
    report['alternative_R_engine'] = 'NEEDS SEPARATE PREREGISTERED FIT' if alt_ok else 'UNSUPPORTED (no estimated upper asymptote in API)'
    (output / 'engine_report.json').write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: {g: v.get(g) for g in ('PASS', 'score_objective', 'score_GH161', 'boundaries', 'vs_python')}
                      for k, v in report['engines'].items()}, indent=1, default=float))
    print('decision', report['decision'], report['alternative_R_engine'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--output', required=True); main(ap.parse_args().output)
