"""Independent-implementation references for DIF, multiplicity, CTT reliability and EAP.

Pre-registered tolerances (set before running; justified by numerics, not by results):
- closed-form statistics vs another library's closed form (MH, CMH, adjustments, alpha): 1e-10;
- two different ML optimisers on the same likelihood (logistic DIF LR): 1e-5 absolute;
- EAP by independent quadrature on the identical grid: 1e-10.
References: statsmodels 0.15 (StratifiedTable, multipletests), scikit-learn (unpenalised
LogisticRegression; lbfgs), pingouin 0.6 (cronbach_alpha). None share code with PsikoAnaliz.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

RNG = np.random.default_rng(20261105)


def _dif_data(n=1200, k=8):
    th = RNG.standard_normal(n); g = RNG.integers(0, 2, n)
    b = np.linspace(-1.2, 1.2, k); shift = np.zeros(k); shift[0] = .6
    x = (RNG.random((n, k)) < expit(th[:, None] - b - shift * g[:, None])).astype(int)
    return x, g


def test_multiplicity_adjustments_match_statsmodels():
    from statsmodels.stats.multitest import multipletests
    from core.dif_classical import adjust_pvalues
    p = np.array([.001, .02, .02, .04, .2, .5, .9, np.nan, 1e-8, .049])
    ok = np.isfinite(p)
    for ours, ref in (('BH', 'fdr_bh'), ('BY', 'fdr_by'), ('Holm', 'holm'), ('Bonferroni', 'bonferroni')):
        got = adjust_pvalues(p, ours)
        np.testing.assert_allclose(got[ok], multipletests(p[ok], method=ref)[1], atol=1e-10, rtol=0)
        assert np.isnan(got[~ok]).all()


def test_mantel_haenszel_matches_statsmodels_stratified_table():
    from statsmodels.stats.contingency_tables import StratifiedTable
    from core.dif_classical import mantel_haenszel
    x, g = _dif_data(); total = x.sum(1)
    for j in (0, 3):
        y = x[:, j]; ours = mantel_haenszel(y, g, total)
        tables = []
        for s in np.unique(total):
            ix = total == s; r, f = ix & (g == 0), ix & (g == 1)
            if r.sum() and f.sum():
                tables.append([[y[r].sum(), r.sum() - y[r].sum()], [y[f].sum(), f.sum() - y[f].sum()]])
        st = StratifiedTable(np.array(tables, float).transpose(1, 2, 0))
        cmh = st.test_null_odds(correction=True)
        assert abs(ours['alpha_mh'] - st.oddsratio_pooled) < 1e-10
        assert abs(ours['se_log_or'] - st.logodds_pooled_se) < 1e-10
        assert abs(ours['chi2'] - cmh.statistic) < 1e-10 and abs(ours['p'] - cmh.pvalue) < 1e-10
    assert mantel_haenszel(x[:, 0], g, total)['delta'] < 0   # sign only: planted DIF against focal -> negative ETS delta


def test_binary_logistic_dif_lr_matches_sklearn():
    from sklearn.linear_model import LogisticRegression
    from core.dif_classical import logistic_dif
    x, g = _dif_data(); total = x.sum(1).astype(float)
    z = (total - total.mean()) / total.std()
    def ll(X, y):
        m = LogisticRegression(penalty=None, solver='lbfgs', tol=1e-12, max_iter=10000).fit(X, y)
        p = np.clip(m.predict_proba(X)[:, 1], 1e-300, 1 - 1e-16)
        return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    for j in (0, 5):
        y = x[:, j]; ours = logistic_dif(y, g, total)
        l0, l1, l2 = (ll(X, y) for X in (z[:, None], np.c_[z, g], np.c_[z, g, z * g]))
        assert abs(ours['lr_uniform'] - max(2 * (l1 - l0), 0)) < 1e-5
        assert abs(ours['lr_nonuniform'] - max(2 * (l2 - l1), 0)) < 1e-5
        assert abs(ours['lr_omnibus'] - max(2 * (l2 - l0), 0)) < 1e-5


def test_cronbach_alpha_matches_pingouin_and_kr20_identity():
    import pingouin as pg
    from core.reliability import compute_reliability_indices
    x, _ = _dif_data(400, 10); frame = pd.DataFrame(x, columns=[f'M{j}' for j in range(10)])
    r = compute_reliability_indices(frame, list(frame.columns))
    ref = pg.cronbach_alpha(data=frame)[0]
    assert abs(r['cronbach_alpha'] - ref) < 1e-6           # project rounds to 6 decimals
    assert abs(r['kr20'] - r['cronbach_alpha']) < 1e-6


def test_undefined_reliability_is_nan_not_zero():
    from core.reliability import compute_reliability_indices
    frame = pd.DataFrame({'a': [1, 0, 1, 0], 'b': [0, 1, 0, 1]})   # every total score = 1
    r = compute_reliability_indices(frame, ['a', 'b'])
    for key in ('cronbach_alpha', 'kr20', 'kr21', 'guttman_lambda4', 'spearman_brown_odd_even'):
        assert np.isnan(r[key]), key


def test_eap_person_estimates_match_independent_quadrature():
    from core.irt_person_abilities import person_abilities
    k = 6; a = np.linspace(.8, 1.6, k); b = np.linspace(-1, 1, k); grid = np.linspace(-4, 4, 81)
    P = expit(a * (grid[:, None] - b))
    curves = pd.DataFrame([dict(Theta=t, Madde=f'M{j}', Kategori=str(c), Olasılık=(P[i, j] if c else 1 - P[i, j]))
                           for i, t in enumerate(grid) for j in range(k) for c in (0, 1)])
    x = RNG.integers(0, 2, (30, k))
    got = person_abilities(x, curves, model='2PL')
    like = np.prod(np.where(x[:, None, :] == 1, P[None], 1 - P[None]), axis=2) * np.exp(-.5 * grid ** 2)
    w = like / like.sum(1, keepdims=True); mean = w @ grid; sd = np.sqrt(w @ grid ** 2 - mean ** 2)
    np.testing.assert_allclose(got['Yetenek (θ)'].to_numpy(float), mean, atol=1e-10, rtol=0)
    np.testing.assert_allclose(got['SH (θ)'].to_numpy(float), sd, atol=1e-10, rtol=0)
