"""Guarded GeneralizIT 0.1.2 adapter. No variance-component estimator here."""
import contextlib
import io
import itertools
import hashlib
from importlib.metadata import version
import sys

import numpy as np
import pandas as pd

DESIGNS = {
    'p x i': ('p', 'i'),
    'p x r': ('p', 'r'),
    'p x i x r': ('p', 'i', 'r'),
    'p x (r:i)': ('p', 'i', 'r'),
    'p x (i:r)': ('p', 'r', 'i'),
}
METHOD = """GeneralizIT 0.1.2; Henderson Method I (ANOVA, REML değil).
Yalnız dengeli, eksiksiz, tek gözlem/hücre ve rastgele yüzeyler.
Ham negatif bileşenler korunur; katsayılarda kütüphanenin sıfırlama politikası uygulanır.
Kütüphane varyansları ve katsayıları 4 ondalığa yuvarlar.
Tamamlayıcı SS/df/MS: dengeli ortogonal altuzaylar için ardışık en küçük kareler;
T, düzeltilmiş SS değildir. Rastgele-etki F/p ve katsayı güven aralığı üretilmez.
D çalışmasında yuvalanmış yüzey sayısı ebeveyn başınadır; birey sayısı sabittir.
Sabit yüzeyler, dengesiz/eksik desenler ve (p:r) x i: NOT VALIDATED / kapalı.
Durum: LIBRARY TESTED; bağımsız doğrulama raporu belirli örneklerle sınırlıdır.
Sonuçlar CERTIFIED olarak etiketlenmez."""


def prepare(data, design, columns, score):
    if design not in DESIGNS:
        raise ValueError('UNSUPPORTED: desen doğrulanmadı.')
    facets = DESIGNS[design]
    selected = [columns[f] for f in facets] + [score]
    if not data.columns.is_unique or len(set(selected)) != len(selected):
        raise ValueError('Her rol için farklı, benzersiz sütun seçin.')
    df = data.loc[:, selected].copy()
    df.columns = list(facets) + ['score']
    if df.isna().any().any():
        raise ValueError('Eksik gözlemler desteklenmiyor; sessiz silme yapılmaz.')
    df['score'] = pd.to_numeric(df['score'], errors='raise')
    if not np.isfinite(df.score).all() or df.score.nunique() < 2:
        raise ValueError('Puanlar sonlu ve değişken olmalıdır.')
    if len(df) > 10000 or any(df[f].nunique() > 100 for f in facets):
        raise ValueError('Bulut sınırı: 10000 gözlem, yüzey başına 100 düzey.')
    if any(df[f].nunique() < 2 for f in facets):
        raise ValueError('Her yüzey en az iki düzey içermelidir.')
    if df.duplicated(list(facets)).any():
        raise ValueError('Tekrarlı hücre: her yüzey birleşimi için tek puan gerekir.')
    # Canonical integer IDs preserve the original mapping separately in settings/data.
    for f in facets:
        df[f] = pd.factorize(df[f], sort=False)[0]
    counts = {f: int(df[f].nunique()) for f in facets}
    if ':' in design:
        child, parent = ('r', 'i') if design == 'p x (r:i)' else ('i', 'r')
        if df.groupby(child)[parent].nunique().max() != 1:
            raise ValueError('Yuvalanmış kimlikler ebeveynler arasında benzersiz olmalıdır.')
        levels = df.groupby(parent)[child].nunique()
        if levels.nunique() != 1 or levels.iloc[0] < 2:
            raise ValueError('Dengesiz yuvalanma veya tek alt düzey desteklenmiyor.')
        counts[child] = int(levels.iloc[0])
    expected = int(np.prod(list(counts.values())))
    if len(df) != expected:
        raise ValueError('Eksik hücre/dengesiz desen desteklenmiyor.')
    return df, counts


def wide_to_long(data, person, items):
    if not items or person in items or len(set(items)) != len(items):
        raise ValueError('Birey ve en az iki farklı madde sütunu seçin.')
    if data[person].isna().any() or data[person].duplicated().any() or len(items) < 2:
        raise ValueError('Geniş veride birey kimliği benzersiz ve eksiksiz olmalıdır.')
    return pd.DataFrame({'p': np.repeat(data[person].to_numpy(), len(items)),
                         'i': np.tile(items, len(data)),
                         'score': data[items].to_numpy().ravel()})


def _errors(engine, levels):
    d = engine.design
    tuples = {k: v for k, v in d.variance_tuple_dictionary.items() if k != 'mean'}
    variances = d.anova_table[['Variance']].drop(index='mean', errors='ignore').clip(lower=0)
    # Pinned, source-verified library methods: no public error-variance result API.
    tau_terms = d._get_tau_facets('p', tuples['p'], tuples)
    tau = float(d._calculate_tau(variances, tau_terms))
    relative = float(d._calculate_little_delta(variances, 'p',
        d._get_little_delta_facets(tau_terms, 'p', tuples['p'], tuples), levels))
    absolute = float(d._calculate_big_delta(variances, 'p',
        d._get_big_delta_facets(tau_terms, tuples), levels))
    if tau <= 0 or relative < 0 or absolute < relative - 1e-12:
        raise ValueError('Tanımlanamayan birey varyansı veya geçersiz hata varyansı; katsayılar kapalı.')
    return {'universe': tau, 'relative_error': relative, 'absolute_error': absolute}


def _anova(df, tuples):
    """Balanced corrected SS/MS for display, separate from Henderson estimates."""
    y = df.score.to_numpy(float)
    y = y - y.mean()
    x = np.ones((len(df), 1))
    previous_sse, previous_rank = float(y @ y), 1
    rows = []
    components = [k for k in tuples if k != 'mean']
    width = 1 + sum(len(df[list(tuples[k])].drop_duplicates()) for k in components[:-1])
    if width > 500 or len(df) * width > 2_000_000:
        raise ValueError('ANOVA bellek sınırı: 500 tasarım sütunu / 2 milyon hücre.')
    for name in components[:-1]:
        keys = pd.Series(list(map(tuple, df[list(tuples[name])].to_numpy())))
        block = pd.get_dummies(keys, dtype=float).to_numpy()
        x = np.column_stack([x, block])
        beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        residual = y - x @ beta
        sse = float(residual @ residual)
        dof = int(rank - previous_rank)
        ss = max(0., previous_sse - sse)
        rows.append((name, ss, dof, ss / dof if dof else np.nan))
        previous_sse, previous_rank = sse, rank
    dof = len(df) - previous_rank
    rows.append((components[-1], previous_sse, dof, previous_sse / dof if dof else np.nan))
    return pd.DataFrame(rows, columns=['source', 'SS', 'df', 'MS']).set_index('source')


def analyze(data, design, columns, score, scenarios=None, fixed_facets=None):
    """Fit one G study and library D scenarios; exact inputs are replayable."""
    from generalizit import GeneralizIT
    from generalizit.g_theory_utils import create_pseudo_df
    if version('generalizit') != '0.1.2':
        raise ValueError('GeneralizIT 0.1.2 gereklidir; farklı sürüm doğrulanmadı.')
    if fixed_facets:
        raise ValueError('NOT VALIDATED: sabit yüzeyler bu ekranda kapalıdır.')
    df, counts = prepare(data, design, columns, score)
    scenarios = {f: [n] for f, n in counts.items()} if scenarios is None else scenarios
    if set(scenarios) != set(counts) or scenarios.get('p') != [counts['p']]:
        raise ValueError('D çalışmasında yüzeyler aynı, birey sayısı sabit olmalıdır.')
    if any(not isinstance(v, list) or not v or any(type(n) is not int or n < 2 or n > 100 for n in v)
           or len(set(v)) != len(v) for v in scenarios.values()):
        raise ValueError('D düzeyleri: benzersiz 2–100 tamsayılar.')
    designs = [dict(zip(scenarios, values)) for values in itertools.product(*scenarios.values())]
    if len(designs) > 25 or any(np.prod(list(s.values())) > 10000 for s in designs):
        raise ValueError('D çalışması sınırı: 25 senaryo ve senaryo başına 10000 hücre.')
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        engine = GeneralizIT(df, design, 'score')
        engine.calculate_anova()
        anova = _anova(df, engine.design.variance_tuple_dictionary)
        engine.calculate_g_coefficients()
        g = engine.design.g_coeffs_table.loc[['p']].copy()
        errors = _errors(engine, engine.design.levels_coeffs)
        engine.calculate_d_study(d_study_design=scenarios)
        rows = []
        for scenario, (label, table) in zip(designs, engine.design.d_study_dict.items()):
            pseudo = create_pseudo_df(scenario, engine.design.variance_tuple_dictionary)
            levels = engine.design._calculate_levels_coeffs(df=pseudo)
            rows.append({**scenario, **table.loc['p'].to_dict(), **_errors(engine, levels)})
    raw = engine.design.anova_table[['T', 'Variance']].drop(index='mean', errors='ignore').copy()
    if not np.isfinite(raw.to_numpy()).all():
        raise ValueError('Sonlu olmayan varyans kestirimi; sonuçlar engellendi.')
    raw['coefficient_variance'] = raw.Variance.clip(lower=0)
    total = raw.coefficient_variance.sum()
    raw['contribution_pct'] = 100 * raw.coefficient_variance / total if total > 0 else np.nan
    return {'anova': anova, 'henderson': engine.design.anova_table.copy(), 'variance': raw,
            'g': g, 'errors': pd.DataFrame([errors], index=['p']), 'd': pd.DataFrame(rows),
            'counts': counts, 'status': 'LIBRARY TESTED', 'method': METHOD, 'log': log.getvalue(),
            'settings': {'design': design, 'columns': columns, 'score': score,
                         'scenarios': scenarios, 'fixed_facets': [], 'generalizit': version('generalizit'),
                         'input_sha256': hashlib.sha256(data.to_csv(index=True).encode()).hexdigest(),
                         'versions': {name: version(name) for name in ('generalizit','numpy','pandas','scipy')},
                         'python': sys.version.split()[0], 'seed': 'Not applicable: deterministic ANOVA'}}


def report_tables(result, target):
    """Read cached tables; retain their original computation in Python replay history."""
    tables = {k: result[k] for k in ('anova', 'henderson', 'variance', 'g', 'errors', 'd')}
    d = tables['d'].copy()
    d['measurement_units_per_person'] = d[[f for f in result['counts'] if f != 'p']].prod(axis=1)
    d['target_met'] = (d['rho^2'] >= target) & (d['phi^2'] >= target)
    tables['d'] = d
    return tables
