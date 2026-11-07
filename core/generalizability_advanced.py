"""Frequentist G-theory extensions. Existing GeneralizIT estimator is untouched.

REML is executed by R lme4, not reimplemented here. Matrix MoM below is limited
to balanced p x one random facet; it estimates cross-dimension covariances.
"""
import hashlib
import itertools
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import sys
from importlib.metadata import version

import numpy as np
import pandas as pd
from scipy import linalg, optimize

from core import generalizability as gt

ROOT = Path(__file__).resolve().parents[1]
LME4_VERSION = '2.1-0'
METHOD = ('R lme4 REML; tüm yüzeyler rastgele, birey ölçme nesnesidir. '
          'REML ile ANOVA/MoM farklı kestiricilerdir. Eksik puan sessiz silinmez. '
          'Gaussian rastgele etkiler ve gözlem seçiminin yok sayılabilirliği varsayılır; '
          'seçici/MNAR eksikliğinde geçerlilik iddiası yoktur. '
          'D katsayıları yeni dengeli hedef desene aittir; dengesiz gözlenen birey '
          'ortalamalarının güvenirliği değildir. Genel sertifika veya rastgele-etki p değeri verilmez.')


def _terms(design):
    if design not in gt.DESIGNS:
        raise ValueError('UNSUPPORTED: bu rastgele yüzey deseni doğrulanmadı.')
    if design == 'p x i':
        return ['p', 'i'], 'p:i'
    if design == 'p x r':
        return ['p', 'r'], 'p:r'
    if design == 'p x i x r':
        return ['p', 'i', 'r', 'p:i', 'p:r', 'i:r'], 'p:i:r'
    parent, child = ('i', 'r') if design == 'p x (r:i)' else ('r', 'i')
    return ['p', parent, f'{parent}:{child}', f'p:{parent}'], f'p:{parent}:{child}'


def structure(data, design, columns, score, fixed_facets=None):
    """Missing observation cells allowed; NA scores, duplicates and confounding are not."""
    if fixed_facets:
        raise ValueError('UNSUPPORTED: sabit yüzeyler için ayrı bilimsel sözleşme gerekir.')
    _terms(design)
    facets = gt.DESIGNS[design]
    selected = [columns[f] for f in facets] + [score]
    if not data.columns.is_unique or len(set(selected)) != len(selected):
        raise ValueError('Her rol için farklı, benzersiz sütun seçin.')
    df = data.loc[:, selected].copy()
    df.columns = [*facets, 'score']
    if df.isna().any().any():
        raise ValueError('Eksik kimlik/puan var; eksik gözlem hücresiyle aynı değildir. Sessiz silme yapılmaz.')
    df['score'] = pd.to_numeric(df.score, errors='raise')
    if not np.isfinite(df.score).all() or df.score.nunique() < 2:
        raise ValueError('Puanlar sonlu ve değişken olmalıdır.')
    if len(df) > 10000 or any(not 2 <= df[f].nunique() <= 100 for f in facets):
        raise ValueError('En çok 10000 gözlem; her yüzeyde 2–100 düzey gerekir.')
    if df.duplicated(list(facets)).any():
        raise ValueError('Tekrarlı gözlem hücresi: ayrı artık yapısı henüz desteklenmiyor.')
    mapping = {}
    for f in facets:
        codes, levels = pd.factorize(df[f], sort=False)
        mapping[f] = [{'label': str(x), 'type': type(x).__name__} for x in levels]
        df[f] = codes
    counts = {f: int(df[f].nunique()) for f in facets}
    if ':' in design:
        parent, child = ('i', 'r') if design == 'p x (r:i)' else ('r', 'i')
        if df.groupby(child)[parent].nunique().max() != 1:
            raise ValueError('Yuvalı alt yüzey kimliği ebeveynler arasında benzersiz olmalıdır.')
        levels = df.groupby(parent)[child].nunique()
        if levels.min() < 2:
            raise ValueError('Her ebeveynde en az iki farklı yuvalı düzey gerekir.')
        counts[child] = int(levels.min())  # explicit NEW balanced D target, not an observed average
    labels = _terms(design)[0]
    codes = [pd.factorize(pd.MultiIndex.from_frame(df[t.split(':')]), sort=False)[0] for t in labels]
    n = len(df)
    sizes = [np.bincount(c).astype(float) for c in codes]
    gram = np.empty((len(codes)+1, len(codes)+1))
    for i, a in enumerate(codes):
        for j, b in enumerate(codes[:i+1]):
            pairs = pd.DataFrame({'a':a,'b':b}).groupby(['a','b']).size()
            aa = pairs.index.get_level_values('a').to_numpy()
            bb = pairs.index.get_level_values('b').to_numpy()
            cross_values = pairs.to_numpy(float)
            value = (cross_values**2).sum() - 2/n*np.sum(sizes[i][aa]*sizes[j][bb]*cross_values) + (sizes[i]@sizes[i])*(sizes[j]@sizes[j])/n**2
            gram[i,j] = gram[j,i] = value
        gram[i,-1] = gram[-1,i] = n-(sizes[i]@sizes[i])/n
    gram[-1,-1] = n-1
    diagonal = np.diag(gram)
    if (diagonal <= 0).any():
        raise ValueError('Tanımlanamayan varyans bileşeni / sabit etkiyle karışma.')
    correlation = gram / np.sqrt(np.outer(diagonal, diagonal))
    eigen = np.linalg.eigvalsh(correlation)
    if eigen[0] <= 0 or eigen[-1]/eigen[0] > 1e10:
        raise ValueError('Varyans bileşenleri karışmış / tanımlanamıyor (çekirdek rank koşulu).')
    return df, counts, {'levels': mapping, 'kernel_min_eigenvalue': float(eigen[0]),
                         'kernel_condition': float(eigen[-1]/eigen[0])}


def _select_engine(data, design, columns, score, engine='auto'):
    if engine not in ('auto', 'ANOVA', 'REML'):
        raise ValueError('Desteklenen motorlar: auto, ANOVA, REML.')
    if engine != 'auto':
        return engine
    try:
        gt.prepare(data, design, columns, score)
        return 'ANOVA'
    except ValueError:
        structure(data, design, columns, score)
        return 'REML'


def _coefficient_row(variance, design, counts):
    value = variance.coefficient_variance
    universe = float(value.loc['p'])
    relative = absolute = 0.
    for name, v in value.items():
        if name == 'p':
            continue
        facets = set(re.findall('[pir]', name))
        divisor = float(np.prod([counts[f] for f in facets if f != 'p']))
        absolute += float(v)/divisor
        if 'p' in facets:
            relative += float(v)/divisor
    if universe <= 0 or relative < 0 or absolute < relative-1e-12:
        raise ValueError('Birey varyansı/hata varyansı geçersiz; katsayılar kapalı.')
    return {'universe': universe, 'relative_error': relative, 'absolute_error': absolute,
            'relative_sem': np.sqrt(relative), 'absolute_sem': np.sqrt(absolute),
            'rho^2': universe/(universe+relative), 'phi^2': universe/(universe+absolute)}


def d_study(result, scenarios):
    if result['status'] not in ('LIBRARY TESTED', 'NUMERICALLY VALIDATED'):
        raise ValueError('Yakınsamayan/tekil modelden D çıkarımı yapılamaz.')
    counts = result['counts']
    if set(scenarios) != set(counts) or scenarios.get('p') != [counts['p']]:
        raise ValueError('Yüzeyler aynı, birey sayısı sabit olmalıdır.')
    if any(not isinstance(v, list) or not v or len(v) != len(set(v)) or
           any(type(x) is not int or not 2 <= x <= 100 for x in v) for v in scenarios.values()):
        raise ValueError('D düzeyleri benzersiz 2–100 tamsayılar olmalıdır.')
    if np.prod([len(v) for v in scenarios.values()]) > 10000:
        raise ValueError('Arama sınırı: en çok 10000 senaryo.')
    rows = []
    for values in itertools.product(*scenarios.values()):
        target = dict(zip(scenarios, values))
        if np.prod(list(target.values())) > 1_000_000:
            raise ValueError('Hedef desen bir milyon gözlem sınırını aşıyor.')
        rows.append({**target, **_coefficient_row(result['variance'], result['settings']['design'], target)})
    return pd.DataFrame(rows)


def analyze(data, design, columns, score, scenarios=None, engine='auto', fixed_facets=None, profile=False):
    if fixed_facets:
        raise ValueError('UNSUPPORTED: sabit yüzey varsayımı doğrulanmadı.')
    chosen = _select_engine(data, design, columns, score, engine)
    if chosen == 'ANOVA':
        if profile:
            raise ValueError('Profil GA yalnız R REML için kullanılabilir.')
        return gt.analyze(data, design, columns, score, scenarios)
    df, counts, validation = structure(data, design, columns, score)
    rscript = shutil.which('Rscript')
    if not rscript:
        raise ValueError('REML çalıştırılmadı: Rscript ve R lme4 2.1-0 gereklidir. ANOVA sessizce ikame edilmez.')
    center, scale = float(df.score.mean()), float(df.score.std(ddof=0))
    with tempfile.TemporaryDirectory(prefix='gtheory-') as tmp:
        folder = Path(tmp)
        df.assign(score=(df.score-center)/scale).to_csv(folder/'data.csv', index=False)
        try:
            run = subprocess.run([rscript, str(ROOT/'scripts/gtheory_reml.R'), str(folder/'data.csv'),
                                  design, str(folder), 'profile' if profile else 'none'],
                                 capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired as exc:
            raise ValueError('R REML/profil 180 saniye bütçesinde bitmedi; sonuç ve çıkarım yok.') from exc
        log = run.stdout+'\nSTDERR\n'+run.stderr
        if run.returncode:
            raise ValueError('R REML başarısız: '+log[-2500:])
        diagnostics = json.loads((folder/'diagnostics.json').read_text())
        diagnostics['standardized_logLik'] = diagnostics['logLik']
        # Restricted likelihood uses N-rank(X), not N; X is the intercept here.
        for name in ('logLik','AIC','BIC'):
            diagnostics[name] += (-1 if name=='logLik' else 2)*(len(df)-1)*np.log(scale)
        diagnostics['likelihood_definition'] = 'REML, N-1 residual contrasts; original score units. AIC/BIC not ML model-comparison statistics.'
        variance = pd.read_csv(folder/'variance.csv').set_index('source')
        variance['Variance'] *= scale**2
        if not np.isfinite(variance.Variance).all() or (variance.Variance < 0).any():
            raise ValueError('Geçersiz R varyans bileşenleri; sonuç engellendi.')
        variance['coefficient_variance'] = variance.Variance
        variance['contribution_pct'] = 100*variance.Variance/variance.Variance.sum()
        effects = pd.read_csv(folder/'random_effects.csv')
        effects['deviation'] *= scale
        effects['conditional_sd'] *= scale
        effects['estimated_universe_score'] = np.where(effects.term=='p',diagnostics['intercept']*scale+center+effects.deviation,np.nan)
        effects['original_label'] = [validation['levels']['p'][int(level)]['label'] if term=='p' else str(level)
                                      for term,level in zip(effects.term,effects.level)]
        ci = pd.read_csv(folder/'profile_ci.csv')
        if len(ci):
            ci[['lower', 'upper']] *= scale**2
    valid = diagnostics['converged'] and not diagnostics['singular'] and diagnostics['hessian_pd']
    status = 'LIBRARY TESTED' if valid else 'NOT CERTIFIED'
    settings = {'design': design, 'columns': columns, 'score': score, 'engine': 'R lme4 REML',
                'profile': profile, 'fixed_facets': [], 'level_map': validation['levels'],
                'input_sha256': hashlib.sha256(data.to_csv(index=True).encode()).hexdigest(),
                'target_note': 'Yeni dengeli D deseni; gözlenen dengesiz ortalamaların katsayısı değil.',
                'versions': {**diagnostics['versions'], 'Python':sys.version.split()[0],
                             **{p:version(p) for p in ('numpy','pandas','scipy')}},
                'score_center': center, 'score_scale': scale,
                'model_mean': diagnostics['intercept']*scale+center,
                'seed': 'Not applicable: deterministic REML', 'convergence': diagnostics,
                'r_script_sha256': hashlib.sha256((ROOT/'scripts/gtheory_reml.R').read_bytes()).hexdigest()}
    result = {'variance': variance, 'counts': counts, 'settings': settings, 'status': status,
              'method': METHOD, 'log': log,
              'diagnostics': pd.DataFrame([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v
                                           for k,v in diagnostics.items()}]),
              'random_effects': effects, 'profile_ci': ci,
              'anova': pd.DataFrame(columns=['SS','df','MS']), 'henderson': pd.DataFrame()}
    targets = scenarios or {f:[n] for f,n in counts.items()}
    settings['scenarios'] = targets
    if valid:
        result['d'] = d_study(result, targets)
        row = _coefficient_row(variance, design, counts)
        result['g'] = pd.DataFrame([{'rho^2':row['rho^2'],'phi^2':row['phi^2']}],index=['p'])
        result['errors'] = pd.DataFrame([{k:v for k,v in row.items() if k not in ('rho^2','phi^2')}],index=['p'])
    else:
        result['g'] = pd.DataFrame([[np.nan,np.nan]], index=['p'], columns=['rho^2','phi^2'])
        result['errors'] = pd.DataFrame(columns=['universe','relative_error','absolute_error'])
        result['d'] = pd.DataFrame(columns=[*counts,'rho^2','phi^2'])
    return result


def optimize_design(result, scenarios, target_g=.8, target_phi=.8, event_cost=1., facet_costs=None, budget=None):
    if not 0 < target_g <= 1 or not 0 < target_phi <= 1 or not np.isfinite(event_cost) or event_cost < 0:
        raise ValueError('Hedef ve maliyetler geçersiz.')
    costs = facet_costs or {}
    if set(costs)-set(result['counts']) or 'p' in costs or any(not np.isfinite(v) or v < 0 for v in costs.values()):
        raise ValueError('Yüzey başına maliyet sonlu, negatif olmayan olmalıdır; birey maliyeti ayrı değildir.')
    if budget is not None and (not np.isfinite(budget) or budget < 0):
        raise ValueError('Bütçe sonlu ve negatif olmayan olmalıdır.')
    d = d_study(result, scenarios)
    facets = [f for f in result['counts'] if f != 'p']
    d['measurement_units_per_person'] = d[facets].prod(axis=1)
    d['cost_per_person'] = event_cost*d.measurement_units_per_person + sum(d[f]*v for f,v in costs.items())
    d['feasible'] = (d['rho^2'] >= target_g) & (d['phi^2'] >= target_phi)
    if budget is not None:
        d['feasible'] &= d.cost_per_person <= budget
    # Cost/G/Phi nondomination. Equal objective triples are retained transparently.
    values = d[['cost_per_person','rho^2','phi^2']].to_numpy()
    pareto = []
    for x in values:
        dominates = (values[:,0] <= x[0]) & (values[:,1] >= x[1]) & (values[:,2] >= x[2])
        strict = (values[:,0] < x[0]) | (values[:,1] > x[1]) | (values[:,2] > x[2])
        pareto.append(not np.any(dominates & strict))
    d['pareto'] = pareto
    feasible = d[d.feasible].sort_values(['cost_per_person','measurement_units_per_person',*facets])
    return {'candidates': d, 'recommendation': feasible.head(1),
            'settings': {'target_g':target_g,'target_phi':target_phi,'event_cost':event_cost,
                         'facet_costs':costs,'budget':budget,'scope':'Only enumerated balanced D targets; no fit rerun'}}


def phi_cut(universe, absolute_error, mean, cut):
    values = np.array([universe, absolute_error, mean, cut], float)
    if not np.isfinite(values).all() or universe <= 0 or absolute_error < 0:
        raise ValueError('Phi-cut için geçerli varyans, ortalama ve kesim gerekir.')
    numerator = universe+(mean-cut)**2
    return numerator/(numerator+absolute_error)


def _psd(matrix):
    scale = float(np.max(np.abs(matrix)))
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12*scale):
        raise ValueError('Kovaryans matrisi sonlu ve simetrik olmalıdır.')
    if np.linalg.eigvalsh(matrix)[0] < -1e-10*scale:
        raise ValueError('PSD olmayan ham kovaryans bileşeni; katsayı hesaplanmaz, kırpma/projeksiyon yapılmaz.')


def multivariate(data, person, facet, dimension, score):
    selected = [person, facet, dimension, score]
    if not data.columns.is_unique or len(set(selected)) != 4:
        raise ValueError('Dört farklı, benzersiz sütun seçin.')
    df = data[selected].copy()
    if df.isna().any().any() or df.duplicated(selected[:3]).any():
        raise ValueError('Boyutlar aynı olaylarda eksiksiz ve tek gözlemle ölçülmelidir.')
    dimensions = list(pd.unique(df[dimension]))
    if not 2 <= len(dimensions) <= 5:
        raise ValueError('Çok değişkenli yol 2–5 boyut içindir.')
    wide = df.pivot(index=[person,facet],columns=dimension,values=score).reindex(columns=dimensions)
    if wide.isna().any().any():
        raise ValueError('Boyutlar arasında eşleşmeyen olay/eksik puan var.')
    first = wide.iloc[:,0].reset_index()
    prepared, counts = gt.prepare(first, 'p x i', {'p':person,'i':facet}, dimensions[0])
    people, items = list(pd.unique(df[person])), list(pd.unique(df[facet]))
    index = pd.MultiIndex.from_product([people, items],names=[person,facet])
    y = wide.reindex(index).to_numpy(float).reshape(len(people),len(items),len(dimensions))
    if not np.isfinite(y).all():
        raise ValueError('Sonlu sayısal puanlar gerekir.')
    n,j,k = y.shape
    mean = y.mean(axis=(0,1)); p = y.mean(axis=1)-mean; f = y.mean(axis=0)-mean
    residual = y-y.mean(axis=1)[:,None,:]-y.mean(axis=0)[None,:,:]+mean
    ms_p = j*(p.T@p)/(n-1); ms_f = n*(f.T@f)/(j-1)
    flat = residual.reshape(-1,k); ms_e = flat.T@flat/((n-1)*(j-1))
    matrices = {'universe':(ms_p-ms_e)/j,'facet':(ms_f-ms_e)/n,'residual':ms_e}
    checks = {name:float(np.linalg.eigvalsh(m)[0]) for name,m in matrices.items()}
    valid = True
    for m in matrices.values():
        try:
            _psd(m)
        except ValueError:
            valid = False
    return {'matrices':matrices,'dimensions':list(map(str,dimensions)),'mean':mean,'counts':counts,
            'status':'LIBRARY TESTED' if valid else 'NOT CERTIFIED','psd_min_eigenvalues':checks,
            'settings':{'person':person,'facet':facet,'dimension':dimension,'score':score,
                        'engine':'balanced multivariate matrix MoM','input_sha256':hashlib.sha256(data.to_csv(index=True).encode()).hexdigest(),
                        'scope':'Balanced p x one random facet only; no separate univariate-fit substitution'}}


def composite(result, n_facet=None, weights=None, allow_negative=False):
    k = len(result['dimensions']); n = result['counts']['i'] if n_facet is None else n_facet
    if type(n) is not int or not 2 <= n <= 100:
        raise ValueError('D yüzey sayısı 2–100 tamsayı olmalıdır.')
    w = np.full(k,1/k) if weights is None else np.array(weights,float)
    if w.shape != (k,) or not np.isfinite(w).all() or not np.isclose(w.sum(),1,rtol=0,atol=1e-10) or (not allow_negative and (w<0).any()):
        raise ValueError('Ağırlıklar sonlu, toplamı 1 olmalı; negatif ağırlık açık izin gerektirir.')
    for m in result['matrices'].values():
        _psd(m)
    u = result['matrices']['universe']; r = result['matrices']['residual']/n
    a = (result['matrices']['facet']+result['matrices']['residual'])/n
    U,R,A = (float(w@m@w) for m in (u,r,a))
    if U <= 0 or R < 0 or A < 0 or np.any(np.diag(u) <= 0):
        raise ValueError('Geçersiz evren/hata varyansı; bileşik katsayı kapalı.')
    dimensions = pd.DataFrame({'dimension':result['dimensions'],'universe':np.diag(u),
        'relative_error':np.diag(r),'absolute_error':np.diag(a),
        'rho^2':np.diag(u)/np.diag(u+r),'phi^2':np.diag(u)/np.diag(u+a)})
    return {'dimension_coefficients':dimensions, 'composite':pd.DataFrame([{'universe':U,'relative_error':R,'absolute_error':A,
            'rho^2':U/(U+R),'phi^2':U/(U+A),'relative_sem':np.sqrt(R),'absolute_sem':np.sqrt(A)}]),
            'universe_covariance':pd.DataFrame(u,index=result['dimensions'],columns=result['dimensions']),
            'relative_error_covariance':pd.DataFrame(r,index=result['dimensions'],columns=result['dimensions']),
            'absolute_error_covariance':pd.DataFrame(a,index=result['dimensions'],columns=result['dimensions']),
            'weights':w,'n_facet':n,'status':result['status']}


def optimize_weights(result, n_facet=None, absolute=False, nonnegative=True):
    base = composite(result, n_facet)
    u = base['universe_covariance'].to_numpy()
    e = base['absolute_error_covariance' if absolute else 'relative_error_covariance'].to_numpy()
    total = u+e
    if np.linalg.eigvalsh(total)[0] <= 0 or np.linalg.cond(total) > 1e8:
        raise ValueError('Ağırlık optimizasyonu için toplam kovaryans pozitif tanımlı ve iyi koşullu olmalıdır.')
    if nonnegative:
        k = len(u)
        run = optimize.minimize(lambda w: -float(w@u@w/(w@total@w)), np.full(k,1/k),
                                method='SLSQP', bounds=[(0,1)]*k,
                                constraints={'type':'eq','fun':lambda w:w.sum()-1},
                                options={'ftol':1e-12,'maxiter':500})
        if not run.success:
            raise ValueError('Ağırlık optimizasyonu yakınsamadı: '+run.message)
        w = run.x; status = 'EXPERIMENTAL / LOCAL OPTIMUM ONLY'
    else:
        eig, vectors = linalg.eigh(u,total); w = vectors[:,-1]
        if abs(w.sum()) <= 1e-10*np.linalg.norm(w):
            raise ValueError('En iyi özvektörün toplamı sıfır; toplamı bir ağırlık olarak temsil edilemez.')
        w = w/w.sum(); status = 'LIBRARY TESTED / GENERALIZED EIGENVALUE'
    return {'weights':w,'coefficient':float(w@u@w/(w@total@w)), 'status':status,
            'nonnegative':nonnegative,'negative_weights':bool((w<0).any()),'absolute':absolute}


def _example(kind='unbalanced'):
    """Deterministic worked data, no random sampling or simulation campaign."""
    rows = []
    for p,i,r in itertools.product(range(8),range(4),range(3)):
        P=(p-3.5)/2; I=i-1.5; R=r-1
        value=40+5*P+2*I+1.5*R+1.2*P*I+.8*P*R+.6*I*R+.4*P*I*R
        rows.append((p,i,r,value))
    d = pd.DataFrame(rows,columns=['p','i','r','score'])
    if kind=='unbalanced':
        return d.drop(index=[0,7,18]).reset_index(drop=True)
    if kind=='balanced':
        return d
    if kind=='multivariate':
        d=d[d.r==0].drop(columns='r').copy()
        P=(d.p-3.5)/2; Q=P**2-np.mean(P**2)
        I=d.i-1.5; J=I**2-np.mean(I**2)
        a=d.assign(score=40+5*P+Q+2*I+J+.2*P*I+.1*Q*J,Dimension='A')
        b=d.assign(score=30+2*P+4*Q+I+2*J+.1*P*J+.2*Q*I)
        b['Dimension']='B'
        return pd.concat([a,b],ignore_index=True)
    raise ValueError('Örnek türü desteklenmiyor.')


# Preparation/example utilities must not be recorded as completed analyses.
select_engine = _select_engine
example = _example
