"""Independent evidence closing documented NOT VERIFIED gaps (validation/SCIENTIFIC_GAP_CLOSURE.md).

Tolerances are fixed before running and justified by numerical error only:
- closed-form vs library probabilities/information: 1e-10 (floating point, no optimisation involved);
- central finite differences (h=1e-6) of an independent formula: relative 1e-6 (O(h^2) + rounding);
- two independent ML engines (native MML vs Python mirt, both stopped at <=1e-8): logLik 1e-6, a 1e-5, b 1e-4;
- GeneralizIT reports components/coefficients to 4 decimals: 5e-5.
Each fit is one small deterministic calibration; no Monte Carlo.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, softmax

from core.irt_library_curves import binary_curves

THETA = np.linspace(-4, 4, 41)


@pytest.fixture(scope='module')
def ordinal_frame():
    rng = np.random.default_rng(7); n, k = 600, 5
    th = rng.standard_normal(n); a = np.linspace(.8, 1.8, k)
    b = np.array([[-1, 0, 1]]) + np.linspace(-.5, .5, k)[:, None]
    x = (rng.random((n, k, 1)) < expit(a[None, :, None] * (th[:, None, None] - b[None]))).sum(2)
    return pd.DataFrame(x, columns=[f'M{j}' for j in range(k)])


def _independent(model, co, t):
    k = len(co)
    if model == 'GRM':
        A = co.discrimination.to_numpy(); B = co[[f'thresholds_{i}' for i in (1, 2, 3)]].to_numpy()
        s = expit(A[None, :, None] * (t[:, None, None] - B[None]))
        ones, zeros = np.ones((len(t), k, 1)), np.zeros((len(t), k, 1))
        return np.concatenate([ones, s], 2) - np.concatenate([s, zeros], 2)
    if model in ('PCM', 'GPCM'):
        A = co.discrimination.to_numpy(); S = co[[f'steps_{i}' for i in (1, 2, 3)]].to_numpy()
        z = A[None, :, None] * (t[:, None, None] - S[None])
        return softmax(np.cumsum(np.concatenate([np.zeros((len(t), k, 1)), z], 2), 2), 2)
    s = co[[f'slopes_{i}' for i in (1, 2, 3, 4)]].to_numpy(); c = co[[f'intercepts_{i}' for i in (1, 2, 3, 4)]].to_numpy()
    return softmax(s[None] * t[:, None, None] + c[None], 2)


@pytest.mark.parametrize('model', ['GRM', 'PCM', 'GPCM', 'NRM'])
def test_polytomous_curves_information_and_sem_match_independent_formulas(ordinal_frame, model):
    from core.irt_b14 import fit_b14
    from core.irt_b14_model_curves import fitted_category_curves
    from core.irt_b14_persons_info import information_table
    result = fit_b14(ordinal_frame, model, categories=[0, 1, 2, 3] if model == 'NRM' else None)
    assert result['result'].converged
    co = result['result'].coef(); k = len(co)
    P = _independent(model, co, THETA)
    shown = fitted_category_curves(result, THETA).pivot_table(
        index=['Theta', 'Madde'], columns='Kategori', values='Olasılık').to_numpy().reshape(len(THETA), k, -1)
    np.testing.assert_allclose(shown, P, atol=1e-10, rtol=0)
    assert np.all((P >= 0) & (P <= 1)) and np.abs(P.sum(2) - 1).max() < 1e-12
    h = 1e-6
    dP = (_independent(model, co, THETA + h) - _independent(model, co, THETA - h)) / (2 * h)
    fisher = (dP ** 2 / P).sum(2)
    table = information_table(result, THETA)
    iif = table[[c for c in table if c.startswith('I_')]].to_numpy()
    assert iif.min() >= 0
    np.testing.assert_allclose(iif, fisher, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(table['Test bilgisi'], iif.sum(1), atol=1e-12, rtol=0)
    np.testing.assert_allclose(table['SEM(θ)'], 1 / np.sqrt(table['Test bilgisi']), atol=1e-12, rtol=0)


def test_binary_1pl_to_4pl_curves_match_independent_derivatives():
    a = np.array([.7, 1.2, 1.9, 1.]); b = np.array([-1., 0., .8, 2.]); c = np.array([0., .2, .1, .05]); d = np.array([1., 1., .95, .9])
    out = binary_curves(THETA, a=a, b=b, c=c, d=d)
    prob = lambda t: c + (d - c) / (1 + np.exp(-a * (t[:, None] - b)))
    P = prob(THETA); h = 1e-6; dP = (prob(THETA + h) - prob(THETA - h)) / (2 * h)
    np.testing.assert_allclose(out['probability'], P, atol=1e-12, rtol=0)
    assert np.all(out['probability'] >= c) and np.all(out['probability'] <= d)
    np.testing.assert_allclose(out['information'], dP ** 2 / (P * (1 - P)), rtol=1e-6, atol=1e-14)
    assert out['information'].min() >= 0
    np.testing.assert_allclose(out['test_information'], out['information'].sum(1), atol=1e-12, rtol=0)


def test_common_free_a_1pl_equals_native_sigma_free_rasch():
    """Reparameterisation identity: logistic(a(z - b)), z~N(0,1)  ==  logistic(t - a b), t~N(0, a^2)."""
    from core.irt_python_mirt_adapter import fit_python_mirt
    from core.rasch import fit_rasch
    rng = np.random.default_rng(11); n, k = 800, 8
    th = rng.normal(0, 1.4, n); b = np.linspace(-1.5, 1.5, k)
    x = (rng.random((n, k)) < expit(th[:, None] - b)).astype(int)
    native = fit_rasch(pd.DataFrame(x, columns=[f'M{j}' for j in range(k)]), points=61)
    py = fit_python_mirt(x, model='1PL', common_discrimination=True, n_quadpts=61, tol=1e-8, max_iter=5000).fit
    assert native['converged'] and py.converged and py.n_parameters == native['param_count']
    co = py.coef(); a = co.discrimination.to_numpy()
    assert np.ptp(a) < 1e-12 and abs(a[0] - native['sd']) < 1e-5
    assert abs(py.log_likelihood - native['loglik']) < 1e-6 and abs(py.aic - native['aic']) < 1e-6
    np.testing.assert_allclose(co.difficulty.to_numpy(), np.asarray(native['b']) / native['sd'], atol=1e-4, rtol=0)


def test_generalizability_p_by_i_matches_independent_anova():
    pytest.importorskip('generalizit')
    from pathlib import Path
    from core.generalizability import analyze
    d = pd.read_csv(Path(__file__).resolve().parents[1] / 'validation/generalizability/balanced.csv')
    x = d.groupby(['p', 'i']).score.mean().reset_index()
    Y = x.pivot(index='p', columns='i', values='score').to_numpy(); n_p, n_i = Y.shape; m = Y.mean()
    ms_p = n_i * ((Y.mean(1) - m) ** 2).sum() / (n_p - 1); ms_i = n_p * ((Y.mean(0) - m) ** 2).sum() / (n_i - 1)
    r = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + m; ms_e = (r ** 2).sum() / ((n_p - 1) * (n_i - 1))
    vp, vi, ve = (ms_p - ms_e) / n_i, (ms_i - ms_e) / n_p, ms_e
    res = analyze(x, 'p x i', {'p': 'p', 'i': 'i'}, 'score', scenarios={'p': [n_p], 'i': [n_i, 10]})
    np.testing.assert_allclose(res['variance']['Variance'].to_numpy(), [vp, vi, ve], atol=5e-5, rtol=0)
    for row, n in zip(res['d'].itertuples(), (n_i, 10)):
        assert abs(row._4 - vp / (vp + ve / n)) < 5e-5 and abs(row._3 - vp / (vp + (vi + ve) / n)) < 5e-5
