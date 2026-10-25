"""Re-evaluate preserved real ML estimates; never refit or certify equivalence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import log_expit, logsumexp, roots_hermite
from validation.r_mirt_phase1.harness import discrepancies

BOUNDS = dict(a=(.1, 5.), b=(-6., 6.), c=(0., .5), d=(.5, 1.))


def evaluate(x, p, theta, weights):
    a, b, c, d = (p[k].to_numpy() for k in ('a', 'b', 'c', 'd'))
    z = (theta[:, None]-b)*a
    with np.errstate(divide='ignore'):
        span = np.log(d-c)
        lp = np.logaddexp(np.log(c), span+log_expit(z))
        lq = np.logaddexp(np.log1p(-d), span+log_expit(-z))
    prob = np.exp(lp)
    log_derivative = span+np.log(a)+log_expit(z)+log_expit(-z)
    info = np.exp(2*log_derivative-lp-lq)
    ll = logsumexp(x @ lp.T + (1-x) @ lq.T + np.log(weights), axis=1).sum()
    return prob, info.sum(axis=1), float(ll)


def audit(source):
    source = Path(source)
    original = json.loads((source/'report.json').read_text())
    nodes, weights = roots_hermite(61); nodes *= np.sqrt(2); weights /= np.sqrt(np.pi)
    result = dict(status='NOT COMPARABLE', refitted=False,
        original_report_sha256=hashlib.sha256((source/'report.json').read_bytes()).hexdigest(),
        software=dict(python=original['python'], packages=original['versions']),
        reason='Physical b bounds imply coupled R intercept/slope constraints; existing box-bound reference cannot express the identical feasible region. No partial alignment is certified.',
        grid=dict(method='Gauss-Hermite N(0,1)', points=61, D=1), models=[])
    for model in ('3PL', '4PL'):
        folder = source/model
        record = next(m for m in original['models'] if m['model']==model)
        files = [folder/'python_parameters.csv', folder/record['reference_directory']/'parameters.csv', folder/'data.csv']
        frames = [pd.read_csv(f) for f in files]
        x = frames[2].to_numpy()
        fits = [f.pivot(index='item', columns='parameter', values='value').loc[frames[2].columns] for f in frames[:2]]
        values = [evaluate(x, p, nodes, weights) for p in fits]
        # ICC/information use the same display grid as the existing harness.
        display = np.linspace(-4, 4, 81)
        curves = [evaluate(x, p, display, np.full(81, 1/81))[:2] for p in fits]
        boundaries = {}
        for engine, p in zip(('python', 'R'), fits):
            boundaries[engine] = [dict(item=item, parameter=k, value=float(v), outside_python_box=bool(v<lo or v>hi))
                for k,(lo,hi) in BOUNDS.items() if k!='d' or model=='4PL'
                for item,v in p[k].items() if v<lo or v>hi or np.isclose(v,lo,atol=1e-6,rtol=0) or np.isclose(v,hi,atol=1e-6,rtol=0)]
        result['models'].append(dict(model=model, status='NOT COMPARABLE',
            original_convergence={k:record[k]['converged'] for k in ('python','R')},
            original_diagnostics={k:record[k] for k in ('python','R')},
            common_grid_loglik=dict(python=values[0][2], R=values[1][2], python_minus_R=values[0][2]-values[1][2]),
            parameters={k:discrepancies(fits[0][k],fits[1][k]) for k in BOUNDS},
            ICC=discrepancies(curves[0][0],curves[1][0]), information=discrepancies(curves[0][1],curves[1][1]),
            boundary_or_outside_python_box=boundaries,
            source_sha256={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}))
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args=p.parse_args(); args.output.write_text(json.dumps(audit(args.source), indent=2, allow_nan=False))
