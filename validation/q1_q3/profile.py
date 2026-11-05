"""Opt-in deterministic constrained profile; existing likelihood/score, no simulation."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import importlib.metadata

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import roots_hermite

from validation.r_mirt_phase1.equivalence_audit import evaluate, BOUNDS
from validation.r_mirt_phase1.m12_diagnostic import scores, feasible_scores

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def gh(n):
    z, w = roots_hermite(n)
    return z*np.sqrt(2), w/np.sqrt(np.pi)


def run(initial, protocol_commit, output, amendment_commit=None):
    protocol = json.loads((HERE/'PROFILE_PROTOCOL.json').read_text())
    data = pd.read_csv(ROOT/protocol['data'])
    assert hashlib.sha256((ROOT/protocol['data']).read_bytes()).hexdigest() == protocol['data_sha256']
    p = pd.read_csv(initial).pivot(index='item', columns='parameter', values='value').loc[data.columns, list(BOUNDS)]
    initial_values = p.to_numpy().ravel()
    archive = np.load(ROOT/'validation/r_mirt_phase1/observed_information/information_matrices.npz')
    labels = [f'{item}:{"u" if k=="d" else k}' for item in p.index for k in p.columns]
    assert archive['labels'].tolist() == labels
    diagonal = np.diag(archive['fine'])
    assert (diagonal > 0).all()
    scale = np.sqrt(diagonal)
    low = np.tile([BOUNDS[k][0] for k in p.columns], len(p))
    high = np.tile([BOUNDS[k][1] for k in p.columns], len(p))
    fixed = labels.index('M12:u')
    free = np.array([j for j in range(len(labels)) if j != fixed])
    lower, upper = (low-initial_values)*scale, (high-initial_values)*scale
    x = data.to_numpy()
    theta, weights = gh(protocol['quadrature']['optimize'])
    audit_theta, audit_weights = gh(protocol['quadrature']['independent_evaluation'])
    result = dict(status='DIAGNOSTIC / NOT CERTIFIED', protocol_commit=protocol_commit,
                  protocol_sha256=hashlib.sha256((HERE/'PROFILE_PROTOCOL.json').read_bytes()).hexdigest(),
                  initial_sha256=hashlib.sha256(Path(initial).read_bytes()).hexdigest(),
                  data_sha256=protocol['data_sha256'], monte_carlo_executions=0,
                  independent_R_optimizer_executed=False,
                  objective_centering_amendment=amendment_commit,
                  versions={k:importlib.metadata.version(k) for k in ('numpy','scipy','pandas','mirt')},
                  reference='Preserved actual R mirt1.48 solution; common validated physical likelihood/score',
                  grid=protocol['grid'], points=[])
    started = time.monotonic()
    options = {k:v for k,v in protocol['optimizer'].items() if k not in ('method','preconditioner')}
    for u in protocol['grid']:
        if time.monotonic()-started >= protocol['limits']['total_seconds']:
            result['stop'] = 'TOTAL TIME BUDGET'; break
        tick = time.monotonic()
        best = {'value':np.inf, 'q':np.zeros(len(free))}

        def decode(q):
            all_q = np.zeros(len(labels)); all_q[free] = q
            values = initial_values+all_q/scale
            # Exact optimizer box endpoints must remain exact physical endpoints.
            values = np.where(all_q==lower, low, np.where(all_q==upper, high, values))
            values[fixed] = u
            return pd.DataFrame(values.reshape(p.shape), index=p.index, columns=p.columns)

        def objective(q):
            if time.monotonic()-tick > protocol['limits']['point_seconds']:
                raise TimeoutError('PER-POINT TIME BUDGET')
            current = decode(q)
            value = -evaluate(x,current,theta,weights)[2] - offset
            gradient = -scores(x,current,theta,weights).to_numpy().ravel()[free]/scale[free]
            if value < best['value']:best.update(value=float(value),q=q.copy())
            return value, gradient

        # Constant centering tightens relative stopping; likelihood/score unchanged.
        offset = -evaluate(x,decode(np.zeros(len(free))),theta,weights)[2] if amendment_commit else 0.
        try:
            fit = minimize(objective, np.zeros(len(free)), jac=True, method='L-BFGS-B',
                           bounds=list(zip(lower[free],upper[free])),options=options)
            q = fit.x
            diagnostics = dict(success=bool(fit.success), message=str(fit.message),
                               iterations=int(fit.nit), evaluations=int(fit.nfev))
        except TimeoutError as exc:
            q = best['q']
            diagnostics = dict(success=False, message=str(exc))
        current = decode(q)
        gradient = scores(x,current,theta,weights)
        audit_gradient = scores(x,current,audit_theta,audit_weights)
        projected = feasible_scores(current,gradient).to_numpy().ravel()
        audit_projected = feasible_scores(current,audit_gradient).to_numpy().ravel()
        raw = float(np.max(np.abs(projected[free])))
        audit_raw = float(np.max(np.abs(audit_projected[free])))
        entry = dict(u=u, optimizer=diagnostics, seconds=time.monotonic()-tick,
                     logLik_GH61=evaluate(x,current,theta,weights)[2],
                     logLik_GH101=evaluate(x,current,audit_theta,audit_weights)[2],
                     nuisance_score_max=raw,nuisance_score_per_person=raw/len(x),
                     audit_nuisance_score_max=audit_raw,audit_score_per_person=audit_raw/len(x),
                     stationary_GH61=bool(diagnostics['success'] and raw<=1e-5),
                     stationary_GH101=bool(diagnostics['success'] and audit_raw<=1e-5),
                     profile_score=float(gradient.loc['M12','d']),
                     M12=current.loc['M12'].to_dict(),
                     boundaries=[labels[j] for j,v in enumerate(current.to_numpy().ravel())
                                 if v==low[j] or v==high[j]],
                     parameters=current.to_dict('index'))
        result['points'].append(entry)
        result['seconds'] = time.monotonic()-started
        Path(output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:entry[k] for k in ('u','seconds','logLik_GH61','nuisance_score_max','stationary_GH61')}),flush=True)
    if len(result['points']) == len(protocol['grid']):result['stop']='GRID COMPLETED'
    result['seconds'] = time.monotonic()-started
    Path(output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--initial',required=True)
    parser.add_argument('--protocol-commit',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--amendment-commit')
    args=parser.parse_args()
    run(args.initial,args.protocol_commit,args.output,args.amendment_commit)
