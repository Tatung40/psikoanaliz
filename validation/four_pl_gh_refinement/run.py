"""Execute PROTOCOL.json once: GH101 bounded-Newton polish of the 4PL fixture MLE, GH161 audit.
Run: PYTHONPATH=. python -m validation.four_pl_gh_refinement.run --protocol-commit <sha>
"""
import argparse, hashlib, importlib.metadata, json, time
from pathlib import Path

import numpy as np
import pandas as pd

from validation.q1_q3.profile import ROOT, gh
from validation.r_mirt_phase1.equivalence_audit import BOUNDS, evaluate
from validation.r_mirt_phase1.information_audit import information
from validation.r_mirt_phase1.m12_diagnostic import feasible_scores, scores

HERE = Path(__file__).resolve().parent
LO = [b[0] for b in BOUNDS.values()]; HI = [b[1] for b in BOUNDS.values()]


def run(commit):
    data_path = ROOT / 'tests/fixtures/irt_b11/data_irt4pl_ok.csv'
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == json.loads((HERE / 'PROTOCOL.json').read_text())['data']['sha256']
    x = pd.read_csv(data_path); X = x.to_numpy()
    src = next(r for r in json.loads((ROOT / 'validation/q1_q3/profile_polished.json').read_text())['points']
               if abs(r['u'] - .6198282615) < 1e-9)
    p0 = pd.DataFrame.from_dict(src['parameters'], orient='index').loc[x.columns, list(BOUNDS)]
    z, w = gh(101); az, aw = gh(161)
    score = lambda v: scores(X, v, z, w)
    value = lambda v: evaluate(X, v, z, w)[2]
    projected = lambda v, g=None: float(np.abs(feasible_scores(v, score(v) if g is None else g).to_numpy()).max())
    p, history, started = p0.copy(), [], time.monotonic()
    for step in range(6):
        g = score(p).to_numpy().ravel(); v = p.to_numpy().ravel()
        free = [j for j, q in enumerate(v) if not ((q == LO[j % 4] and g[j] <= 0) or (q == HI[j % 4] and g[j] >= 0))]
        h, _ = information(score, p, float(np.cbrt(np.finfo(float).eps))); h = (h + h.T) / 2
        face = h[np.ix_(free, free)]; eig = np.linalg.eigvalsh(face); cond = float(np.linalg.cond(face))
        rec = dict(step=step, score=projected(p), logLik=value(p), information_min=float(eig[0]), condition=cond); history.append(rec)
        if eig[0] <= 0 or cond > 1e8: rec['stop'] = 'INFORMATION GATE'; break
        if rec['score'] <= 1e-5: rec['stop'] = 'STATIONARY'; break
        d = np.zeros(len(v)); d[free] = np.linalg.solve(face, g[free]); old = rec['logLik']; ok = False
        for k in range(10):
            nv = v + d * .5 ** k
            if any(nv[j] < LO[j % 4] or nv[j] > HI[j % 4] for j in range(len(nv))): continue
            cand = pd.DataFrame(nv.reshape(p.shape), index=p.index, columns=p.columns)
            if value(cand) >= old and projected(cand) < rec['score']:
                p, ok = cand, True; rec['accepted_backtrack'] = k; break
        if not ok: rec['stop'] = 'NO MONOTONE SCORE-IMPROVING STEP'; break
    lls = {f'GH{n}': evaluate(X, p, *gh(n))[2] for n in (41, 61, 101, 161)}
    audit = projected(p, scores(X, p, az, aw)); final = projected(p)
    out = dict(protocol_commit=commit, status='DIAGNOSTIC / NOT CERTIFIED', monte_carlo_executions=0,
               independent_R_optimizer_executed=False, seconds=time.monotonic() - started,
               versions={k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'pandas', 'mirt')},
               history=history, logLik=lls, score_GH101=final, score_GH161_audit=audit,
               gate_stationary_GH101=bool(final <= 1e-5),
               gate_audit_GH161=bool(audit <= 1e-5 and abs(lls['GH161'] - lls['GH101']) <= 1e-3),
               max_abs_change_vs_GH61_start=float((p - p0).abs().to_numpy().max()),
               M12=p.loc['M12'].to_dict(), parameters=p.to_dict('index'),
               boundaries=[f'{i}:{k}' for i in p.index for k in p.columns if p.loc[i, k] in BOUNDS[k]])
    (HERE / 'result.json').write_text(json.dumps(out, indent=2) + '\n')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--protocol-commit', required=True)
    r = run(ap.parse_args().protocol_commit)
    print(json.dumps({k: r[k] for k in ('seconds', 'logLik', 'score_GH101', 'score_GH161_audit', 'gate_stationary_GH101',
                                        'gate_audit_GH161', 'max_abs_change_vs_GH61_start', 'M12', 'boundaries')}, indent=1))
    print([(h['step'], round(h['score'], 9), h.get('stop')) for h in r['history']])
