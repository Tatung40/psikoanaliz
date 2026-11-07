"""Deterministic mathematical, real-engine and independent native R checks. No MC."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

import numpy as np
import pandas as pd
import pytest

from core import generalizability as gt
from core import generalizability_advanced as ga

ROOT=Path(__file__).resolve().parents[1]


def fixture(design,missing=False):
    d=ga.example('balanced')
    if design=='p x i':d=d[d.r==0].drop(columns='r')
    if design=='p x r':d=d[d.i==0].drop(columns='i')
    if design=='p x (r:i)':d=d.assign(r=d.i*3+d.r)
    if design=='p x (i:r)':d=d.assign(i=d.r*4+d.i)
    d=d.reset_index(drop=True)
    return d.drop(index=[0,7,18]).reset_index(drop=True) if missing else d


@pytest.mark.parametrize('design',gt.DESIGNS)
def test_design_selection_and_kernel_identification(design):
    cols={f:f for f in gt.DESIGNS[design]}
    for missing in (False,True):
        d=fixture(design,missing)
        prepared,counts,info=ga.structure(d,design,cols,'score')
        assert len(prepared)==len(d) and info['kernel_condition']<1e10
        assert ga.select_engine(d,design,cols,'score')==('REML' if missing else 'ANOVA')
    assert ga.select_engine(d,design,cols,'score','REML')=='REML'


@pytest.mark.parametrize('change',['NA','duplicate','nested','confounded','fixed','constant'])
def test_invalid_and_confounded_fail_before_r(change):
    design='p x (r:i)' if change=='nested' else 'p x i'
    d=fixture(design);cols={f:f for f in gt.DESIGNS[design]}
    kw={}
    if change=='NA':d.loc[0,'score']=np.nan
    if change=='duplicate':d=pd.concat([d,d.iloc[:1]])
    if change=='nested':d.r=d.r%3
    if change=='confounded':d=d[d.p==d.i]
    if change=='fixed':kw['fixed_facets']=['i']
    if change=='constant':d.score=1
    with pytest.raises(ValueError):ga.structure(d,design,cols,'score',**kw)


def test_missing_runtime_is_explicit_and_anova_preserved(monkeypatch):
    d=fixture('p x i');cols={'p':'p','i':'i'}
    original=gt.analyze(d,'p x i',cols,'score')
    actual=ga.analyze(d,'p x i',cols,'score')
    for name in ('variance','g','errors','d','budget'):
        pd.testing.assert_frame_equal(original[name],actual[name])
    monkeypatch.setattr(ga.shutil,'which',lambda _:None)
    with pytest.raises(ValueError,match='REML çalıştırılmadı'):
        ga.analyze(d.iloc[1:],'p x i',cols,'score')


def test_matrix_covariance_composite_scaling_and_singular_rejection():
    d=ga.example('multivariate');r=ga.multivariate(d,'p','i','Dimension','score')
    assert r['status']=='LIBRARY TESTED' and abs(r['matrices']['universe'][0,1])>0
    base=ga.composite(r,4,[.25,.75]);more=ga.composite(r,8,[.25,.75])
    assert more['composite'].iloc[0]['phi^2']>base['composite'].iloc[0]['phi^2']
    moved=ga.multivariate(d.assign(score=100+.001*d.score),'p','i','Dimension','score')
    for k,m in r['matrices'].items():np.testing.assert_allclose(moved['matrices'][k]/1e-6,m,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(ga.composite(moved,4,[.25,.75])['composite'][['rho^2','phi^2']],base['composite'][['rho^2','phi^2']],atol=1e-10)
    broken={**r,'matrices':{**r['matrices'],'universe':np.array([[1,2],[2,1.]])}}
    with pytest.raises(ValueError,match='PSD'):ga.composite(broken)
    with pytest.raises(ValueError,match='eşleşmeyen'):ga.multivariate(d.iloc[1:],'p','i','Dimension','score')
    with pytest.raises(ValueError):ga.composite(r,weights=[-1,2])
    assert ga.composite(r,weights=[-1,2],allow_negative=True)['weights'][0]==-1


def test_weight_objectives_phi_cut_and_d_search_are_analytical():
    r=ga.multivariate(ga.example('multivariate'),'p','i','Dimension','score')
    for absolute in (False,True):
        opt=ga.optimize_weights(r,4,absolute,nonnegative=False)
        c=ga.composite(r,4,opt['weights'],allow_negative=True)['composite'].iloc[0]
        assert abs(opt['coefficient']-c['phi^2' if absolute else 'rho^2'])<1e-10
        limited=ga.optimize_weights(r,4,absolute,nonnegative=True)
        assert 'LOCAL' in limited['status'] and min(limited['weights'])>=0
        assert limited['coefficient']<=opt['coefficient']+1e-10
    assert ga.phi_cut(4,1,10,10)==.8
    assert ga.phi_cut(4,1,10,0)==104/105
    d=fixture('p x i');fit=gt.analyze(d,'p x i',{'p':'p','i':'i'},'score')
    search=ga.optimize_design(fit,{'p':[8],'i':list(range(2,32))},.8,.8,event_cost=2,facet_costs={'i':3},budget=100)
    table=search['candidates'];assert len(table)==30  # separate search expands the 25-scenario fit limit
    np.testing.assert_allclose(table.cost_per_person,5*table.i)
    assert table.loc[table.i==4,'rho^2'].iloc[0]==pytest.approx(fit['g'].iloc[0]['rho^2'],abs=5e-5)
    for _,row in table.iterrows():
        dominates=(table.cost_per_person<=row.cost_per_person)&(table['rho^2']>=row['rho^2'])&(table['phi^2']>=row['phi^2'])
        strict=(table.cost_per_person<row.cost_per_person)|(table['rho^2']>row['rho^2'])|(table['phi^2']>row['phi^2'])
        assert bool(row.pareto)==(not (dominates&strict).any())


def _r():
    r=shutil.which('Rscript')
    if not r:
        if os.getenv('REQUIRE_GT_ADV_R')=='1':pytest.fail('Native R required; not executed')
        pytest.skip('NOT EXECUTED: Rscript unavailable')
    return r


def test_r_version_pin_uses_numeric_version_not_printed_punctuation():
    bridge=(ROOT/'scripts/gtheory_reml.R').read_text()
    reference=(ROOT/'validation/gtheory_advanced/reference.R').read_text()
    assert "package_version('2.1-0')" in bridge and "package_version('2.1-0')" in reference
    assert "as.character(utils::packageVersion('lme4')) !=" not in bridge


@pytest.mark.parametrize('design',gt.DESIGNS)
@pytest.mark.parametrize('missing',[False,True])
def test_independent_r_reml_mapping_and_balanced_mom(design,missing,tmp_path):
    r=_r();d=fixture(design,missing);cols={f:f for f in gt.DESIGNS[design]}
    actual=ga.analyze(d,design,cols,'score',engine='REML')
    for name in ('variance','g','errors','d','diagnostics','random_effects','profile_ci'):
        actual[name].to_csv(tmp_path/('python_'+name+'.csv'))
    (tmp_path/'settings.json').write_text(json.dumps(actual['settings'],indent=2))
    (tmp_path/'adapter.log').write_text(actual['log'])
    normalized,_,_=ga.structure(d,design,cols,'score');scale=float(normalized.score.std(ddof=0))
    normalized.score=(normalized.score-normalized.score.mean())/scale
    normalized.to_csv(tmp_path/'normalized.csv',index=False)
    done=subprocess.run([r,str(ROOT/'validation/gtheory_advanced/reference.R'),str(tmp_path/'normalized.csv'),design,str(tmp_path/'reference')],capture_output=True,text=True,timeout=180)
    (tmp_path/'reference.stdout').write_text(done.stdout);(tmp_path/'reference.stderr').write_text(done.stderr)
    assert done.returncode==0,done.stderr
    reference=pd.read_csv(tmp_path/'reference/variance.csv')
    key=lambda s:frozenset(s.split(':'))
    expected={key(row.source):row.Variance*scale**2 for row in reference.itertuples()}
    diff={name:float(row.Variance-expected[key(name)]) for name,row in actual['variance'].iterrows()}
    tol=1e-8*max(1,max(expected.values()))
    assert max(abs(x) for x in diff.values())<=tol
    diag=actual['settings']['convergence']
    direct=pd.read_csv(tmp_path/'reference/fit.csv').iloc[0]
    assert diag['converged']==(int(direct.convergence_messages)==0)
    assert diag['singular']==bool(direct.singular)
    if not diag['converged'] or diag['singular'] or not diag['hessian_pd']:
        assert actual['status']=='NOT CERTIFIED'
        assert actual['g'].isna().all().all() and actual['d'].empty
    assert abs(diag['standardized_logLik']-direct.logLik)<1e-8
    assert abs(diag['logLik']-(direct.logLik-(len(d)-1)*np.log(scale)))<1e-8
    coefficient_status='SUPPRESSED: singular/nonconverged/invalid information'
    if actual['status']=='LIBRARY TESTED':
        counts=actual['counts'];scenarios={f:[n] if f=='p' else [n,n*2] for f,n in counts.items()}
        predictions=ga.d_study(actual,scenarios)
        for ref in pd.read_csv(tmp_path/'reference/coefficients.csv').itertuples():
            selected=predictions
            for f,n in counts.items():selected=selected[selected[f]==(n if f=='p' else n*ref.multiplier)]
            row=selected.iloc[0]
            for name in ('universe','relative_error','absolute_error'):
                assert row[name]==pytest.approx(getattr(ref,name)*scale**2,rel=1e-8,abs=tol)
            for name in ('relative_sem','absolute_sem'):
                assert row[name]==pytest.approx(getattr(ref,name)*scale,rel=1e-8,abs=tol)
            assert row['rho^2']==pytest.approx(ref.rho,abs=1e-8)
            assert row['phi^2']==pytest.approx(ref.phi,abs=1e-8)
        coefficient_status='NUMERICALLY VALIDATED: balanced target G/D, SEM, error variances'
    # Balanced-interior REML and ANOVA have matching EMS solutions; no assertion for boundary/unbalanced.
    mom_status='NOT COMPARABLE: missing cells or variance boundary'
    if not missing and actual['status']=='LIBRARY TESTED':
        mom=gt.analyze(d,design,cols,'score')
        if (mom['variance'].Variance>0).all():
            reference_mom={frozenset(__import__('re').findall('[pir]',str(name))):row.Variance for name,row in mom['variance'].iterrows()}
            np.testing.assert_allclose([row.Variance for _,row in actual['variance'].iterrows()],
                [reference_mom[key(name)] for name in actual['variance'].index],rtol=2e-4,atol=2e-4*max(1,max(reference_mom.values())))
            mom_status='NUMERICALLY VALIDATED: balanced interior only'
    (tmp_path/'comparison.json').write_text(json.dumps({'status':'NUMERICALLY VALIDATED','scope':'adapter vs direct R lmer on identical standardized data',
        'variance_differences':diff,'converged':diag['converged'],'singular':diag['singular'],'mom_status':mom_status,
        'coefficients_status':coefficient_status,
        'fixture_sha256':hashlib.sha256(d.to_csv(index=True).encode()).hexdigest(),'versions':actual['settings']['versions']},indent=2))


def test_independent_r_multivariate_polarization(tmp_path):
    r=_r();d=ga.example('multivariate');d.to_csv(tmp_path/'input.csv',index=False)
    done=subprocess.run([r,str(ROOT/'validation/gtheory_advanced/multivariate_reference.R'),str(tmp_path/'input.csv'),str(tmp_path/'reference')],capture_output=True,text=True,timeout=60)
    (tmp_path/'R.stdout').write_text(done.stdout);(tmp_path/'R.stderr').write_text(done.stderr)
    assert done.returncode==0,done.stderr
    actual=ga.multivariate(d,'p','i','Dimension','score');deltas={}
    for name,m in actual['matrices'].items():
        ref=pd.read_csv(tmp_path/f'reference/{name}.csv',index_col=0).to_numpy()
        np.testing.assert_allclose(m,ref,rtol=1e-10,atol=1e-10*max(1,abs(ref).max()))
        deltas[name]=float(abs(m-ref).max())
    table=pd.read_csv(tmp_path/'reference/coefficients.csv')
    for row in table.itertuples():
        w=[row.w1,1-row.w1];c=ga.composite(actual,int(row.n),w)['composite'].iloc[0]
        for mine,theirs in [('universe','universe'),('relative_error','relative_error'),('absolute_error','absolute_error'),('rho^2','rho'),('phi^2','phi'),('relative_sem','relative_sem'),('absolute_sem','absolute_sem')]:
            assert c[mine]==pytest.approx(getattr(row,theirs),rel=1e-10,abs=1e-10)
        mean=float(np.array(w)@actual['mean'])
        for cut,expected in [(0,row.cut0),(40,row.cut40)]:
            assert ga.phi_cut(c.universe,c.absolute_error,mean,cut)==pytest.approx(expected,abs=1e-10)
    (tmp_path/'comparison.json').write_text(json.dumps({'status':'NUMERICALLY VALIDATED','scope':'balanced two-dimensional p x i fixture, weights and D counts only','matrix_max_differences':deltas},indent=2))


def test_r_profile_and_affine_and_singular_diagnostics(tmp_path):
    _r();d=fixture('p x i');cols={'p':'p','i':'i'}
    base=ga.analyze(d,'p x i',cols,'score',engine='REML',profile=True)
    assert base['status']=='LIBRARY TESTED' and len(base['profile_ci'])==3
    assert (base['profile_ci'].lower>=0).all() and (base['profile_ci'].upper>=base['profile_ci'].lower).all()
    for row in base['profile_ci'].itertuples():
        assert row.lower <= base['variance'].loc[row.source,'Variance'] <= row.upper
    moved=ga.analyze(d.assign(score=100+.001*d.score),'p x i',cols,'score',engine='REML')
    np.testing.assert_allclose(moved['variance'].Variance/1e-6,base['variance'].Variance,rtol=2e-4,atol=2e-4*base['variance'].Variance.max())
    np.testing.assert_allclose(moved['g'],base['g'],atol=2e-4)
    zero=d.assign(score=d.p.astype(float))
    # Degenerate residual/no facet variability must not produce usable G/Phi.
    try:
        singular=ga.analyze(zero,'p x i',cols,'score',engine='REML')
    except ValueError as exc:
        assert 'R REML başarısız' in str(exc) or 'Geçersiz' in str(exc)
    else:
        assert singular['status']=='NOT CERTIFIED' and singular['g'].isna().all().all() and singular['d'].empty
    for name in ('variance','profile_ci','diagnostics'):base[name].to_csv(tmp_path/(name+'.csv'))
    (tmp_path/'R.log').write_text(base['log'])
