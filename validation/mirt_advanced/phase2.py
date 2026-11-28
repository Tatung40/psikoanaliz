"""Advanced MIRT Phase II CI harness (predeclared; unchanged stage-1 criteria; one run per case, no retries).

A. Fixed-asymptote M3PL/M4PL through the native R backend.
   - Adapter fidelity: compared with the archived direct R mirt fits (phase3 expansion_evidence.json).
   - Engine independence: compared with the production SciPy engine (core.irt_multidim_general).
B. GRM/GPCM bifactor through the Python adapter (core.mirt_advanced), compared with native R mirt on a
   common grid using the stage-1 harness. The fixtures (seeds below) were declared before any fit.
C. GRM EFA 2D explicit high-precision profile, compared with a fresh native R fit; runtimes measured.
D. Binary bifactor 4D diagnostic: R bfactor and Python bfactor dimension reduction, quadpts 21, same 1000-iteration budget.
E. R bfactor graded/gpcm (dimension reduction) on the B fixtures. Informational only, because the integration rule differs.
"""
import argparse, hashlib, json, sys, time, traceback
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'validation/mirt_stage1'))
import compare as S1  # noqa: E402  stage-1 harness, read-only
from validation.r_mirt_phase1.harness import discrepancies  # noqa: E402
from core import mirt_r_backend as RB, mirt_advanced as ADV, irt_multidim_general as SG  # noqa: E402
from mirt.estimation.quadrature import GaussHermiteQuadrature  # noqa: E402

CRIT = json.loads((ROOT / 'validation/mirt_stage1/criteria.json').read_text())
P3 = ROOT / 'validation/mirt_phase3'
SEEDS = {'GRM_Bifactor_4D': 20261127, 'GPCM_Bifactor_4D': 20261128}


def ok(checks): return all(v['status'] == 'PASS' for v in checks.values())


def archived_fixed():
    ev = json.loads((P3 / 'expansion_evidence.json').read_text())
    return [c for s in ev['stages'] if s['name'] in ('B_fixed_expansion', 'C_fixed') for c in s['cases']]


def part_a():
    out = []
    for case in archived_fixed():
        st = case['settings']; fam, q = st['family'], st['q']; pattern = np.array(st['pattern'])
        rec = dict(family=fam, q=q, within=st['within'], c=st['c'], u=st['u'])
        try:
            data = pd.read_csv(P3 / f'fixtures/2PL_{q}D.csv')
            theta = np.vstack([np.zeros(q), np.eye(q), -np.eye(q), np.linspace(-3, 3, 31)[:, None] * np.ones(q)])
            r = RB.fit(data, f'M{fam}_fixed', pattern=pattern, c=st['c'], u=st['u'] if fam == '4PL' else None,
                       n_quadpts=15, max_iter=3000, tol=1e-8, trace=theta)
            spec = SG.Specification(pattern, fam, True, False, False, c=st['c'], u=st['u'])
            t = time.perf_counter(); py = SG.fit(data, spec, n_quadpts=15); py_s = time.perf_counter() - t
            v = r['values']; items = list(data.columns)
            A = np.array([[v[(v.item == it) & (v.name == f'a{d+1}')].value.iloc[0] for d in range(q)] for it in items])
            d_ = np.array([v[(v.item == it) & (v.name == 'd')].value.iloc[0] for it in items])
            eta = v[v.name.str.startswith('eta')].value.to_numpy()
            rx = np.concatenate([A[spec.pattern], d_, eta]); x = py['parameters']
            g = np.array([v[(v.item == it) & (v.name == 'g')].value.iloc[0] for it in items])
            uu = np.array([v[(v.item == it) & (v.name == 'u')].value.iloc[0] for it in items])
            rp = np.asarray(r['trace_probabilities'])[..., 1].T
            rI = np.array([[np.reshape(m, (q, q)) for m in item] for item in r['trace_information']]).transpose(1, 0, 2, 3)
            mu, pcov = SG.eap(py['objective'], x)
            arch = case['R']
            rec['adapter_fidelity'] = dict(
                logLik_vs_archived_direct_R=discrepancies([r['logLik']], [arch['logLik']], CRIT['native_common_logLik']),
                covariance_vs_archived_direct_R=discrepancies(r['covariance'], arch['covariance'], CRIT['calibration_covariance']),
                iterations=dict(backend=r['iterations'], archived=arch['iterations']))
            rec['independent_vs_SciPy'] = dict(
                parameters=discrepancies(x, rx, CRIT['calibration_parameters']),
                covariance=discrepancies(spec.covariance(spec.unpack(x)[-1]), r['covariance'], CRIT['calibration_covariance']),
                probabilities=discrepancies(SG.probabilities(spec, x, theta), rp, CRIT['calibration_probability']),
                information=discrepancies(SG.information(spec, x, theta), rI, CRIT['calibration_information']),
                common_logLik=discrepancies([-py['objective'].evaluate(x)[0]], [r['common_grid_logLik']], CRIT['common_logLik']),
                R_native_vs_common_grid=discrepancies([r['logLik']], [r['common_grid_logLik']], CRIT['native_common_logLik']),
                EAP=discrepancies(mu, r['EAP'], CRIT['calibration_EAP']),
                posterior_SD=discrepancies(np.sqrt(np.diagonal(pcov, axis1=1, axis2=2)), r['posterior_SD'], CRIT['calibration_posterior_SD']))
            rec['fixed_asymptotes_exact'] = bool(np.allclose(g, st['c'], rtol=0, atol=1e-12) and np.allclose(uu, st['u'] if fam == '4PL' else 1, rtol=0, atol=1e-12))
            rec['n_parameters'] = dict(R=r['n_parameters'], SciPy=spec.n_parameters, equal=int(r['n_parameters']) == spec.n_parameters)
            rec['converged'] = dict(R=r['converged'], SciPy=py['converged'])
            rec['seconds'] = dict(R_backend_wall=r['wall_seconds'], SciPy=py_s)
            fid = all(rec['adapter_fidelity'][k]['status'] == 'PASS' for k in ('logLik_vs_archived_direct_R', 'covariance_vs_archived_direct_R'))
            rec['status'] = ('NUMERICALLY VALIDATED / NOT CERTIFIED' if fid and ok(rec['independent_vs_SciPy']) and rec['fixed_asymptotes_exact']
                             and rec['n_parameters']['equal'] and r['converged'] and py['converged'] else 'FAIL')
        except Exception as e:
            rec['status'] = 'BLOCKED'; rec['blocker'] = repr(e); rec['traceback'] = traceback.format_exc()[-1500:]
        out.append(rec)
    return out


def bifactor_profile(name):
    model = name.split('_')[0]; groups = np.repeat(np.arange(3), 3); k = 9; q = 4
    pattern = ADV.bifactor_pattern(groups)
    syntax = '\n'.join(f'F{j+1} = ' + ','.join(str(i + 1) for i in np.flatnonzero(pattern[:, j])) for j in range(q))
    return dict(name=name, model=model, q=q, k=k, pattern=pattern.tolist(), correlated=False, covariance_mode='pair',
                exploratory=False, groups=None, syntax=syntax, N=600, seed=SEEDS[name], n_quadpts=7, max_iter=1000, tol=1e-7,
                criteria_sha256=hashlib.sha256((ROOT / 'validation/mirt_stage1/criteria.json').read_bytes()).hexdigest())


HP = {'flag': False}


def adapter_fit(data, model, pattern, *, correlated=False, exploratory=False, labels=None, start_values=None, n_quadpts=7, max_iter=1000, tol=1e-7):
    assert not correlated and not exploratory
    groups = np.argmax(np.asarray(pattern)[:, 1:], 1)
    return ADV.fit_polytomous_bifactor(data, groups, model, n_quadpts=n_quadpts, max_iter=max_iter, tol=tol, start_values=start_values, high_precision=HP['flag'])


def part_b(out, names=tuple(SEEDS)):
    original_profile, original_fit = S1.profile, S1.fit
    res = {}
    try:
        S1.profile = lambda n: bifactor_profile(n) if n in SEEDS else original_profile(n)
        S1.fit = adapter_fit
        for name in names:
            t = time.perf_counter(); r = S1.run(name, out / name)
            res[name] = dict(status=r['status'], blocker=r.get('blocker'), seconds=time.perf_counter() - t,
                             Python=r.get('Python'), R={k: r.get('R', {}).get(k) for k in ('converged', 'iterations', 'logLik')},
                             failed=[k for k, v in r.get('comparison', {}).items() if v['status'] != 'PASS'],
                             max_abs={k: v.get('max_absolute') for k, v in r.get('comparison', {}).items()}, high_precision=HP['flag'])
    finally:
        S1.profile, S1.fit = original_profile, original_fit
    return res


def part_c(out):
    name = 'GRM_EFA_2D'; d = out / name
    base = S1.run(name, d)
    rec = dict(baseline_status=base['status'], baseline_failed=[k for k, v in base.get('comparison', {}).items() if v['status'] != 'PASS'])
    try:
        cfg = json.loads((d / 'settings.json').read_text()); data = pd.read_csv(d / 'data.csv')
        hp = ADV.fit_grm_high_precision(data, np.array(cfg['pattern']), exploratory=True, n_quadpts=cfg['n_quadpts'])
        grid = GaussHermiteQuadrature(cfg['n_quadpts'], cfg['q'])
        r = S1.compare_pair(dict(fit=hp['fit'], covariance=hp['covariance']), d / 'fit', cfg, data, grid, dict(profile=name), CRIT)
        rec.update(high_precision_status=r['status'], failed=[k for k, v in r['comparison'].items() if v['status'] != 'PASS'],
                   max_abs={k: v.get('max_absolute') for k, v in r['comparison'].items()}, diagnostics=hp['diagnostics'])
    except Exception as e:
        rec.update(high_precision_status='BLOCKED', blocker=repr(e))
    return rec


def part_d(out):
    import mirt
    d = out / 'Bifactor_4D_DR'; cfg, data, _, _ = S1.inputs('Bifactor_4D', d)
    rec = dict(settings=dict(quadpts=21, max_iter=1000, tol=1e-7, N=len(data), integration='dimension reduction (both engines)'))
    try:
        r = RB.fit(data, 'bfactor', groups=np.array(cfg['groups']), itemtype='2PL', n_quadpts=21, max_iter=1000, tol=1e-7,
                   trace=np.vstack([np.zeros(4), np.eye(4), -np.eye(4)]))
        v = r['values']; items = list(data.columns)
        RA = np.array([[v[(v.item == it) & (v.name == f'a{j+1}')].value.iloc[0] for j in range(4)] for it in items])
        rec['R'] = dict(converged=r['converged'], iterations=r['iterations'], logLik=r['logLik'], seconds=r['seconds'])
        t = time.perf_counter()
        f = mirt.bfactor(data.to_numpy(), cfg['groups'], n_quadpts=21, max_iter=1000, tol=1e-7, compute_standard_errors=False)
        PA = f.model.get_loading_matrix(); sign = np.sign(PA.sum(0)) * np.sign(RA.sum(0)); PA = PA * sign
        rec['Python'] = dict(converged=bool(f.converged), iterations=int(f.n_iterations), logLik=float(f.log_likelihood), seconds=time.perf_counter() - t)
        rec['loadings_max_abs_diff_sign_aligned'] = float(np.max(abs(PA - RA)))
        rec['loading_structure_valid'] = bool(np.all((RA != 0) == (np.asarray(S1.profile('Bifactor_4D')['pattern']) != 0)))
        rec['logLik_abs_diff_note'] = 'Different quadrature rules (R equally spaced vs Python GH): informational, no gate'
        rec['logLik_abs_diff'] = abs(rec['R']['logLik'] - rec['Python']['logLik'])
    except Exception as e:
        rec['blocker'] = repr(e)
    return rec


def part_e(out):
    res = {}
    for name, it in (('GRM_Bifactor_4D', 'graded'), ('GPCM_Bifactor_4D', 'gpcm')):
        try:
            data = pd.read_csv(out / 'B' / name / 'data.csv')
            r = RB.fit(data, 'bfactor', groups=np.repeat(np.arange(3), 3), itemtype=it, n_quadpts=21, max_iter=1000, tol=1e-7)
            res[name] = dict(converged=r['converged'], iterations=r['iterations'], logLik=r['logLik'], seconds=r['seconds'], n_parameters=r['n_parameters'])
        except Exception as e:
            res[name] = dict(blocker=repr(e))
    return res


def part_f(out):
    """Iteration 2 (GRM bifactor only): same fixture, same criteria, explicit high-precision continuation."""
    HP['flag'] = True
    try:
        return part_b(out, ('GRM_Bifactor_4D',))
    finally:
        HP['flag'] = False


def main(out, only=None):
    out.mkdir(parents=True, exist_ok=True); summary = dict(criteria_sha256=hashlib.sha256((ROOT / 'validation/mirt_stage1/criteria.json').read_bytes()).hexdigest(),
                                                           engine=RB.engine_available())
    for key, fn in (('A_fixed_asymptote', part_a), ('B_poly_bifactor', lambda: part_b(out / 'B')),
                    ('C_grm_high_precision', lambda: part_c(out / 'C')), ('D_binary_bifactor_DR', lambda: part_d(out / 'D')),
                    ('E_R_poly_bfactor_DR', lambda: part_e(out)), ('F_grm_bifactor_high_precision', lambda: part_f(out / 'F'))):
        if only and key not in only:
            continue
        t = time.perf_counter()
        try:
            summary[key] = fn()
        except Exception as e:
            summary[key] = dict(blocker=repr(e), traceback=traceback.format_exc()[-2000:])
        print(key, f'{time.perf_counter() - t:.1f}s', flush=True)
        (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
    print(json.dumps(summary, indent=1, default=str))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); p.add_argument('--only', nargs='*')
    a = p.parse_args(); main(a.output, a.only)
