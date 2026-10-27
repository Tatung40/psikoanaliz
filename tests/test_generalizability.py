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
    np.testing.assert_allclose(r['variance'].Variance, np.round([(msp-mse)/ni,(msi-mse)/np_,mse],4),atol=1e-12)


@pytest.mark.parametrize('change', ['missing','cell','duplicate','constant','one','nested_ids','unbalanced'])
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
    # One unit in library output is 1e-4; compare already-rounded components exactly
    # up to CSV/binary conversion, NOT a discrepancy-calibrated tolerance.
    np.testing.assert_allclose(actual['variance'].Variance,reference.loc[actual['variance'].index,'Variance'],rtol=0,atol=1e-12)
    anova=pd.read_csv(out/'anova.csv').set_index('source')
    np.testing.assert_allclose(actual['anova'],anova.loc[actual['anova'].index,['SS','df','MS']],rtol=1e-10,atol=1e-10)
    expected=pd.read_csv(out/'coefficients.csv')
    deltas={}
    for idx,row in expected.iterrows():
        measured=actual['d'].iloc[0 if idx==0 else -1]
        for key in ('universe','relative_error','absolute_error'):
            np.testing.assert_allclose(measured[key],row[key],rtol=1e-12,atol=1e-12)
            deltas[f'{idx}_{key}']=float(measured[key]-row[key])
        assert measured['rho^2']==row.rho and measured['phi^2']==row.phi
    (tmp_path/'comparison.json').write_text(json.dumps({'design':design,'status':'NUMERICALLY VALIDATED',
        'scope':'Only this balanced deterministic fixture; random facets; G and two D scenarios',
        'differences':deltas,'generalizit':'0.1.2',
        'fixture_sha256':hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        'python_settings':actual['settings']},indent=2))
