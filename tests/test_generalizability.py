"""No simulations. Real GeneralizIT, deterministic fixture and independent references."""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile
import hashlib

import numpy as np
import pandas as pd
import pytest

from core import generalizability as gt

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'validation/generalizability/balanced.csv'


def data_for(design):
    df = pd.read_csv(FIXTURE)
    if design == 'p x i':
        df = df[df.r == 0].drop(columns='r')
    if design == 'p x r':
        df = df[df.i == 0].drop(columns='i')
    if design == 'p x (r:i)':
        df.r = df.i * 2 + df.r
    if design == 'p x (i:r)':
        df.i = df.r * 3 + df.i
    return df.reset_index(drop=True)


def calculation(design):
    data = data_for(design)
    columns = {f: f for f in gt.DESIGNS[design]}
    _, counts = gt.prepare(data, design, columns, 'score')
    scenarios = {f: [n] if f == 'p' else [n, n*2] for f, n in counts.items()}
    return gt.analyze(data, design, columns, 'score', scenarios)


@pytest.mark.parametrize('design', gt.DESIGNS)
def test_real_library_g_d_and_corrected_anova(design):
    result = calculation(design)
    # SS decomposition and df are identities, no empirical cutoff.
    y = data_for(design).score
    np.testing.assert_allclose(result['anova'].SS.sum(), ((y-y.mean())**2).sum(), rtol=1e-12)
    assert result['anova']['df'].sum() == len(y)-1
    assert (result['anova']['df'] > 0).all()
    g = result['g'].loc['p']
    e = result['errors'].loc['p']
    assert g['rho^2'] == round(e.universe/(e.universe+e.relative_error),4)
    assert g['phi^2'] == round(e.universe/(e.universe+e.absolute_error),4)
    d = result['d'].iloc[0]
    assert d['rho^2'] == g['rho^2'] and d['phi^2'] == g['phi^2']
    assert result['d'].iloc[-1]['rho^2'] >= g['rho^2']
    assert result['status'] == 'LIBRARY TESTED'


def test_independent_balanced_two_way_expected_mean_squares():
    d = data_for('p x i'); np_, ni = d.p.nunique(), d.i.nunique()
    # Independent closed-form balanced ANOVA reference, not GeneralizIT.
    y = d.pivot(index='p',columns='i',values='score').to_numpy()
    mean=y.mean(); p=y.mean(axis=1); i=y.mean(axis=0)
    msp=ni*((p-mean)**2).sum()/(np_-1)
    msi=np_*((i-mean)**2).sum()/(ni-1)
    mse=((y-p[:,None]-i[None,:]+mean)**2).sum()/((np_-1)*(ni-1))
    r=calculation('p x i')
    ref = np.array([(msp-mse)/ni,(msi-mse)/np_,mse])
    np.testing.assert_allclose(r['variance'].Variance, ref, rtol=0, atol=1e-9*max(1, abs(ref).max()))


@pytest.mark.parametrize('change', ['missing','cell','duplicate','constant','one','nested_ids','unbalanced','inf','text'])
def test_invalid_designs_fail_closed(change):
    ds='p x (r:i)' if change in ('nested_ids','unbalanced') else 'p x i'
    d=data_for(ds)
    if change=='missing': d.loc[0,'score']=np.nan
    if change=='cell': d=d.iloc[1:]
    if change=='duplicate': d=pd.concat([d,d.iloc[:1]])
    if change=='constant': d['score']=1
    if change=='one': d=d[d.i==0]
    if change=='nested_ids': d.r=d.r%2
    if change=='unbalanced': d=d[d.r!=0]
    if change=='inf': d.loc[0,'score']=np.inf
    if change=='text': d['score']=d.score.astype(object); d.loc[0,'score']='x'
    with pytest.raises(ValueError):
        gt.analyze(d,ds,{f:f for f in gt.DESIGNS[ds]},'score')


def test_fixed_and_unsupported_and_scenario_limits():
    d=data_for('p x i'); c={'p':'p','i':'i'}
    with pytest.raises(ValueError,match='sabit'):
        gt.analyze(d,'p x i',c,'score',fixed_facets=['i'])
    with pytest.raises(ValueError,match='UNSUPPORTED'):
        gt.analyze(d,'(p:r) x i',c,'score')
    for scenarios in ({'p':[7],'i':[3]}, {'p':[6],'i':[1]}, {'p':[6],'i':[2]*2},
                      {'p':[6],'i':list(range(2,29))}):
        with pytest.raises(ValueError):gt.analyze(d,'p x i',c,'score',scenarios)


def test_negative_components_preserved_and_clipping_explicit():
    d=data_for('p x (i:r)')
    # Remove the r main effect: negative sampling component remains visible.
    d.score=d.score-3*d.r
    r=gt.analyze(d,'p x (i:r)',{'p':'p','i':'i','r':'r'},'score')
    assert (r['variance'].Variance < 0).any()
    assert (r['variance'].coefficient_variance >= 0).all()
    assert 'sıfırlama' in r['method']


def test_wide_conversion_preserves_labels_values_and_missing():
    d=pd.DataFrame({'id':['a','b'],'p':[1,3],'score':[2,np.nan]})
    long=gt.wide_to_long(d,'id',['p','score'])
    assert long.i.tolist()==['p','score','p','score']
    assert long.p.tolist()==['a','a','b','b']
    assert pd.isna(long.score.iloc[-1])
    with pytest.raises(ValueError):gt.wide_to_long(pd.concat([d,d]),'id',['p','score'])


def test_exports_and_python_replay(tmp_path):
    from ui.generalizability_view import exports, _zip
    from core.reproducible import Recorder, python_code, bundle
    result=calculation('p x i')
    _,files,report=exports(result,.8)
    assert files['variance.png'].startswith(b'\x89PNG')
    assert 'CERTIFIED' in report and 'REML değil' in report
    assert pd.read_excel(io.BytesIO(files['G_D.xlsx']),sheet_name='d').shape[0]==2
    with zipfile.ZipFile(io.BytesIO(_zip(files))) as z:
        assert {'engine.log','settings.json','anova.csv','D.png'} <= set(z.namelist())
    recorder=Recorder('G theory')
    recorder.invoke(gt.analyze,(data_for('p x i'),'p x i',{'p':'p','i':'i'},'score'),{})
    manifest=recorder.manifest()
    assert manifest['status']=='GENERATED / UNVERIFIED'
    code=python_code(manifest)
    with zipfile.ZipFile(io.BytesIO(bundle(manifest,code))) as z:z.extractall(tmp_path)
    done=subprocess.run([os.sys.executable,'analysis.py'],cwd=tmp_path,capture_output=True,text=True)
    assert done.returncode==0,done.stderr[-1000:]


@pytest.mark.parametrize('design', gt.DESIGNS)
def test_independent_r_anova_and_g_d(design,tmp_path):
    r=shutil.which('Rscript')
    if not r:
        if os.getenv('REQUIRE_GTHEORY_R')=='1':pytest.fail('R execution required, Rscript missing')
        pytest.skip('NOT EXECUTED: Rscript unavailable')
    out=tmp_path/'r'
    cmd=[r,str(ROOT/'validation/generalizability/reference.R'),str(FIXTURE),design,str(out)]
    run=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
    (tmp_path/'stdout.log').write_text(run.stdout)
    (tmp_path/'stderr.log').write_text(run.stderr)
    assert run.returncode==0,run.stderr
    actual=calculation(design)
    for name in ('anova','henderson','variance','g','errors','d'):
        actual[name].to_csv(tmp_path/f'python_{name}.csv')
    (tmp_path/'python_settings.json').write_text(json.dumps(actual['settings'],indent=2))
    reference=pd.read_csv(out/'variance.csv').set_index('source')
    # PROTOCOL_FINAL.json: unrounded reference; library rounding acts on rescaled scores only.
    tol=1e-9*max(1,reference.raw_variance.abs().max())
    np.testing.assert_allclose(actual['variance'].Variance,reference.loc[actual['variance'].index,'raw_variance'],rtol=0,atol=tol)
    anova=pd.read_csv(out/'anova.csv').set_index('source')
    np.testing.assert_allclose(actual['anova'],anova.loc[actual['anova'].index,['SS','df','MS']],rtol=1e-10,atol=1e-10)
    expected=pd.read_csv(out/'coefficients.csv')
    deltas={}
    for idx,row in expected.iterrows():
        measured=actual['d'].iloc[0 if idx==0 else -1]
        for key in ('universe','relative_error','absolute_error'):
            np.testing.assert_allclose(measured[key],row[key],rtol=0,atol=tol)
            deltas[f'{idx}_{key}']=float(measured[key]-row[key])
        for mine,theirs in (('rho^2','rho'),('phi^2','phi')):
            assert abs(measured[mine]-row[theirs])<=5e-5+1e-9   # library 4-dp display rounding only
            deltas[f'{idx}_{mine}']=float(measured[mine]-row[theirs])
    (tmp_path/'comparison.json').write_text(json.dumps({'design':design,'status':'NUMERICALLY VALIDATED',
        'scope':'Only this balanced deterministic fixture; random facets; G and two D scenarios',
        'differences':deltas,'generalizit':'0.1.2',
        'fixture_sha256':hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        'python_settings':actual['settings']},indent=2))


@pytest.mark.parametrize('design', gt.DESIGNS)
def test_affine_invariance_gd_identity_sem_and_budget(design):
    """Regression for the library 4-dp rounding defect: results must not depend on score units/location."""
    d = data_for(design); cols = {f: f for f in gt.DESIGNS[design]}
    base = gt.analyze(d, design, cols, 'score')
    moved = gt.analyze(d.assign(score=100 + .001 * d.score), design, cols, 'score')
    pd.testing.assert_frame_equal(base['g'], moved['g'])
    np.testing.assert_allclose(moved['variance'].Variance / 1e-6, base['variance'].Variance, rtol=1e-9, atol=1e-9)
    e = base['errors'].iloc[0]; budget = base['budget'].sum()
    np.testing.assert_allclose([budget.relative, budget.absolute], [e.relative_error, e.absolute_error], rtol=1e-12)
    np.testing.assert_allclose([e.relative_sem, e.absolute_sem], np.sqrt([e.relative_error, e.absolute_error]), rtol=1e-15)
    first = base['d'].iloc[0]                                   # D study at the original counts = G study
    assert first['rho^2'] == base['g'].loc['p', 'rho^2'] and first['phi^2'] == base['g'].loc['p', 'phi^2']
    np.testing.assert_allclose(first[['universe', 'relative_error', 'absolute_error']].to_numpy(float),
                               e[['universe', 'relative_error', 'absolute_error']].to_numpy(float), rtol=1e-12)


@pytest.mark.parametrize('design', ['p x i', 'p x r', 'p x i x r'])
def test_independent_gtheoryr_crossed(design, tmp_path):
    src = os.getenv('GTHEORYR_SRC'); r = shutil.which('Rscript')
    if not (src and r):
        if os.getenv('REQUIRE_GTHEORY_R') == '1': pytest.fail('gtheoryr comparison required')
        pytest.skip('NOT EXECUTED: GTHEORYR_SRC/Rscript unavailable')
    out = tmp_path / 'g'
    run = subprocess.run([r, str(ROOT / 'validation/generalizability/gtheoryr_compare.R'), src, str(FIXTURE), design, str(out)],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    ref = pd.read_csv(out / 'gtheoryr.csv'); actual = calculation(design)
    vc = pd.read_csv(out / 'components.csv').set_index('source')
    tol = 1e-9 * max(1, vc.value.abs().max())
    np.testing.assert_allclose(actual['variance'].Variance.to_numpy(), vc.value.to_numpy(), rtol=0, atol=tol)
    an = pd.read_csv(out / 'anova.csv').set_index('source')
    np.testing.assert_allclose(actual['anova'][['SS', 'df', 'MS']].to_numpy(), an[['SS', 'df', 'MS']].to_numpy(), rtol=1e-10, atol=1e-10)
    if (vc.value >= 0).all():                                    # gtheoryr does not truncate negative components
        for i, row in ref.iterrows():
            m = actual['d'].iloc[0 if i == 0 else -1]
            np.testing.assert_allclose(m[['relative_error', 'absolute_error']].to_numpy(float), [row.relative_error, row.absolute_error], rtol=0, atol=tol)
            assert abs(m['rho^2'] - row.g) <= 5e-5 + 1e-9 and abs(m['phi^2'] - row.phi) <= 5e-5 + 1e-9
    (tmp_path / 'comparison.json').write_text(json.dumps(dict(design=design, reference='gtheoryr 0.2.0 @3b53991',
        max_component_diff=float((actual['variance'].Variance.to_numpy() - vc.value.to_numpy()).__abs__().max()),
        coefficients_compared=bool((vc.value >= 0).all())), indent=2))
