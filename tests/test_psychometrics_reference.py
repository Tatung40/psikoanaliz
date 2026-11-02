"""Deterministic published-example checks for linking, equating, reliability and rotation.

Reference numbers (STUIRT, POLYEQUATE, Equating Recipes, R psych, R GPArotation,
R mirt) were transcribed from the Apache-2.0 psychometrics test suite
(github.com/meyerjp3/psychometrics); no Java source code is used.
No fitting, simulation or randomness except one small LSAT7 calibration.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from core.factor_rotation import gpa_oblique
from core.irt_equating import observed_score_equating, true_score_equating
from core.irt_linking import characteristic_curve_linking, mean_mean, mean_sigma, to_y_scale
from core.reliability_extended import feldt_brennan, feldt_gilmer, guttman_lambdas, reliability_family

F = json.loads((Path(__file__).parent / 'fixtures/psychometrics_reference/reference_values.json').read_text())


@pytest.mark.parametrize('method,key', [('stocking_lord', 'stocking_lord_Q1Q2'), ('haebara', 'haebara_Q1Q2')])
def test_characteristic_curve_linking_matches_stuirt(method, key):
    L = F['linking']; e = L['expected'][key]
    r = characteristic_curve_linking(L['x'], L['y'], L['points'], L['x_density'], L['points'], L['y_density'],
                                     method=method, part='Q1Q2', D=L['D'])
    assert abs(r['A'] - e['A']) <= e['tol'] and abs(r['B'] - e['B']) <= e['tol']


def test_haebara_backward_criterion_matches_reference():
    H = F['haebara']; e = H['expected']['Q1']
    r = characteristic_curve_linking(H['x'], H['y'], H['points'], H['x_density'], H['points'], H['y_density'],
                                     method='haebara', part='Q1', D=H['D'])
    assert abs(r['A'] - e['A']) <= e['tol'] and abs(r['B'] - e['B']) <= e['tol']


def test_linking_recovers_exact_known_transformation_and_identity():
    L = F['linking']; x = np.array(L['x']); y = to_y_scale(x, 1.3, -.4)
    for method in ('stocking_lord', 'haebara'):
        r = characteristic_curve_linking(x, y, L['points'], L['x_density'], method=method)
        assert abs(r['A'] - 1.3) < 1e-7 and abs(r['B'] + .4) < 1e-7
    for f in (mean_sigma, mean_mean):
        r = f(x, y); assert abs(r['A'] - 1.3) < 1e-12 and abs(r['B'] + .4) < 1e-12
    with pytest.raises(ValueError):
        characteristic_curve_linking(x, y[:-1], L['points'], L['x_density'])


def test_true_score_equating_matches_polyequate():
    T = F['true_score']; r = true_score_equating(T['x'], T['y'], D=T['D'])
    ref = np.array(T['x_theta']); solvable = np.abs(ref) < 99
    np.testing.assert_allclose(r['y_equivalent'], T['y_equivalent'], atol=T['tol'], rtol=0)
    np.testing.assert_allclose(r['theta'][solvable], ref[solvable], atol=T['tol'], rtol=0)
    assert np.isnan(r['theta'][~solvable]).all()  # no ±99 sentinel presented as an estimate


def test_observed_score_equating_matches_equating_recipes():
    O = F['observed_score']; qx, qy = np.array(O['x_dist']).T, np.array(O['y_dist']).T
    r = observed_score_equating(O['x'], (qx[0], qx[1]), O['y'], (qy[0], qy[1]), w1=O['w1'], D=O['D'])
    np.testing.assert_allclose(r['y_equivalent'], O['y_equivalent'], atol=O['tol'], rtol=0)
    assert abs(r['x_distribution'].sum() - 1) < 1e-12 and abs(r['y_distribution'].sum() - 1) < 1e-12


def test_guttman_lambdas_match_psych_and_feldt_identities():
    G = F['guttman']; g = guttman_lambdas(G['matrix'])
    for key in ('L1', 'L2', 'L3', 'L5', 'L6'):
        assert abs(g['lambda' + key[1]] - G['expected'][key]) < 1e-6
    assert g['lambda1'] <= g['lambda3'] <= g['lambda2']
    cs = np.full((5, 5), .4) + np.eye(5) * .6           # compound symmetry: all classical coefficients = alpha
    alpha = guttman_lambdas(cs)['lambda3']
    assert abs(feldt_brennan(cs) - alpha) < 1e-12 and abs(feldt_gilmer(cs) - alpha) < 1e-12
    assert np.isnan(feldt_gilmer(cs[:2, :2]))
    assert reliability_family(cs)['status'].endswith('NOT CERTIFIED')
    with pytest.raises(ValueError):
        guttman_lambdas([[1, 2], [3, 1]])


@pytest.mark.parametrize('method', ['oblimin', 'quartimin', 'geomin'])
def test_gpa_oblique_matches_gparotation(method):
    G = F['gpa']; A = np.array(G['loadings']); r = gpa_oblique(A, method)
    assert r['converged']
    np.testing.assert_allclose(r['loadings'], G[method], atol=1e-4, rtol=0)
    np.testing.assert_allclose(r['loadings'] @ r['phi'] @ r['loadings'].T, A @ A.T, atol=1e-10)
    if method == 'geomin':
        np.testing.assert_allclose(r['phi'], G['geomin_phi'], atol=1e-4, rtol=0)


def test_python_mirt_2pl_sx2_matches_r_mirt_on_lsat7():
    import mirt
    from core.irt_library_diagnostics import library_itemfit
    S = F['lsat7']
    x = np.array([[int(v) for v in k.split(',')] for k, n in S['patterns'].items() for _ in range(n)])
    assert x.shape == (1000, 5)
    fit = mirt.fit_mirt(x, model='2PL', n_quadpts=61, tol=1e-8, max_iter=5000, verbose=False)
    assert fit.converged
    table = mirt.itemfit(fit, x, statistics=['S_X2'], min_expected=0.)
    np.testing.assert_allclose(table.S_X2.to_numpy(), S['S_X2'], atol=S['tol'], rtol=0)
    np.testing.assert_array_equal(table.df.to_numpy(), S['df'])
    shown = library_itemfit(fit, x, model='2PL')
    assert 'LSAT7' in shown['Bilimsel durum'].iloc[0] and 'NOT CERTIFIED' in shown['Bilimsel durum'].iloc[0]
