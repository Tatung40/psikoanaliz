"""Real public-library fits and fail-closed UI/export regression."""
from tests.reproducible_ui import scientific_downloads
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
import mirt
import numpy as np
import pandas as pd
import pytest
from core.python_mirt_dif import MODELS,analyze,prepare,fingerprint,gate,support_matrix
from reports.python_mirt_dif_report import downloads

@pytest.fixture(scope='module')
def binary():
    d=pd.read_csv('tests/fixtures/irt_b10/data_dif2pl_mixed.csv')
    return d

@pytest.fixture(scope='module')
def known(binary):
    items=[c for c in binary if c!='Grup']
    result=analyze(binary,items=items,group='Grup',reference='R',focal='F',model='2PL',
        anchors=[c for c in items if c not in ('M3','M7')],studied=['M3','M7'],
        parameters=('intercepts','discrimination'),n_quadpts=41,tol=1e-5)
    import os
    if os.environ.get('PSIKO_DIF_EVIDENCE'):
        from validation.python_mirt_dif_check import save_snapshot
        save_snapshot(result,Path(os.environ['PSIKO_DIF_EVIDENCE'])/'known.json')
    return result


def test_known_dif_actual_library_and_report(known):
    fits=known['fits'];t=known['tables']
    assert t['Fit'].Converged.all()
    for row in t['DIF'].itertuples():
        assert row.LR==2*(fits['H1'].log_likelihood-fits['H0:'+row.Item].log_likelihood)
        assert row.df==2 and row.LR>0
        from mirt.multigroup.invariance import invariance_lrt
        assert invariance_lrt(fits['H0:'+row.Item],fits['H1'])['p_value']<.05
    effects=t['Contrasts']
    b=effects[(effects.Item=='M3') & (effects.Parameter=='difficulty')].Focal_minus_reference.item()
    a=effects[(effects.Item=='M7') & (effects.Parameter=='discrimination')].Focal_minus_reference.item()
    assert b>0 and a<0
    assert t['DIF'][['p','q']].isna().all().all()
    assert (t['DIF'].Decision=='ÇIKARIM KAPALI').all()
    assert known['metadata']['Status']=='NOT CERTIFIED'
    bundle=downloads(known,'M3')
    assert bundle['png'].startswith(b'\x89PNG')
    for frame in (pd.read_csv(io.BytesIO(bundle['csv'])),pd.read_excel(io.BytesIO(bundle['xlsx']),sheet_name='DIF')):
        assert frame[['p','q']].isna().all().all()
    with zipfile.ZipFile(io.BytesIO(bundle['zip'])) as z:
        assert json.loads(z.read('settings.json'))['Inference'] is False
        assert 'NOT CERTIFIED' in z.read('report.md').decode()
        assert pd.read_csv(z.open('DIF.csv'))[['p','q']].isna().all().all()


@pytest.mark.parametrize('model',MODELS)
def test_actual_model_api_and_nominal_safeguards(model):
    base='1PL' if model=='Rasch' else model
    x=mirt.simdata(model=base,n_persons=100,n_items=4,n_categories=3,seed=94)
    d=pd.DataFrame(np.concatenate([x,x]),columns=list('ABCD'));d['G']=['R']*100+['F']*100
    labels=None
    if model in ('GRM','PCM','GPCM'):d[list('ABCD')]+=1
    if model=='NRM':
        labels=('C','A','B');d[list('ABCD')]=d[list('ABCD')].replace(dict(enumerate(labels)))
    result=analyze(d,items=list('ABCD'),group='G',reference='R',focal='F',model=model,
        anchors=['A','B','C'],studied=['D'],labels=labels,max_iter=80,n_quadpts=15)
    assert np.isfinite(result['tables']['Fit'].LogLik).all()
    assert result['tables']['DIF'][['p','q']].isna().all().all()
    assert result['tables']['DIF'].df.item()>0
    if model=='NRM':assert 'DTF' not in result['tables']
    if model in ('GRM','PCM','GPCM'):assert result['metadata']['Category_labels']==('1','2','3')
    p=result['tables']['Curves'];np.testing.assert_allclose(p.groupby(['Group','Item','Theta']).Probability.sum(),1)
    I=result['tables']['IIF'];T=result['tables']['TIF']
    np.testing.assert_allclose(I.groupby(['Group','Theta']).Information.sum(),T.Information)


def test_gates_and_complete_fingerprinting(binary):
    good=SimpleNamespace(log_likelihood=-100,n_parameters=12,converged=True,standard_errors={})
    bad=SimpleNamespace(log_likelihood=-101,n_parameters=14,converged=False,standard_errors={'x':{0:np.array([-1.])}})
    lr,df,why=gate(good,bad,'4PL')
    assert lr==-2 and df==2 and all(s in why for s in ('NOT CONVERGED','Geçersiz LR','Geçersiz SE','3PL/4PL'))
    key=fingerprint(binary,('x',));changed=binary.copy();changed.iloc[0,1]=1-changed.iloc[0,1]
    assert key!=fingerprint(changed,('x',)) and key!=fingerprint(binary,('y',))
    matrix=support_matrix().set_index('Model')
    assert not matrix.loc['RSM','Calibration'] and not matrix.Wald.any() and not matrix.Inferential_p.any()
    items=list(binary.columns[1:5]);bad=binary.copy();bad.loc[0,items[0]]=np.nan
    with pytest.raises(ValueError):prepare(bad,items,'Grup','R','F','2PL')
    bad=binary.copy();bad.loc[bad.Grup=='F',items[0]]=0
    with pytest.raises(ValueError,match='tüm kategoriler'):prepare(bad,items,'Grup','R','F','2PL')
    with pytest.raises(ValueError,match='Sabit a'):analyze(binary,items=items,group='Grup',reference='R',focal='F',model='1PL',anchors=items[:2],studied=items[2:],parameters=('discrimination',))


def test_ui_real_flow_stale_and_downloads(binary,monkeypatch):
    from streamlit.testing.v1 import AppTest
    import ui.python_mirt_dif_view as view
    calls=[]
    original=view.analyze
    def actual(*args,**kwargs):
        calls.append(kwargs);return original(*args,**kwargs)
    monkeypatch.setattr(view,'analyze',actual)
    # Small fixed subset; engine still executes, no mocked fit.
    d=binary.groupby('Grup',sort=False).head(100).copy();items=list(d.columns[1:5]);d=d[['Grup']+items]
    at=AppTest.from_file(str(Path('app.py').resolve()),default_timeout=120)
    for k,v in dict(raw_df=d,scored_df=None,item_cols=items,id_col=None,group_col='Grup',nav_page='MTK: Python mirt DIF').items():at.session_state[k]=v
    at.run();assert not at.exception
    at.selectbox(key='pmd_model').set_value('2PL').run()
    at.button(key='pmd_run').click().run();assert not at.exception and len(calls)==1
    assert len(scientific_downloads(at))==6
    assert 'pmd_download_academic_docx' in {b.key for b in scientific_downloads(at)}
    assert {b.key for b in at.get('download_button')} - {b.key for b in scientific_downloads(at)} == {'repro_py','repro_zip'}
    at.run();assert not at.exception and len(calls)==1 and len(scientific_downloads(at))==6
    old=at.session_state[view.KEY]
    mutations=[('pmd_alpha',.04),('pmd_correction','holm'),('pmd_anchors',items[:3]),
               ('pmd_model','Rasch'),('pmd_test','Omnibus'),('pmd_group',items[0]),('pmd_reference','F'),('pmd_items',items[:3]),
               ('pmd_solver_max_iter',100),('pmd_solver_n_quadpts',31),('pmd_solver_tol',1e-5),
               ('pmd_sensitivity',True)]
    for control,value in mutations:
        # Each route reuses the already measured fit; no additional calibrations.
        fresh=AppTest.from_file(str(Path('app.py').resolve()),default_timeout=120)
        for k,v in dict(raw_df=d,scored_df=None,item_cols=items,id_col=None,group_col='Grup',nav_page='MTK: Python mirt DIF').items():fresh.session_state[k]=v
        fresh.run();fresh.selectbox(key='pmd_model').set_value('2PL').run();fresh.session_state[view.KEY]=dict(old);fresh.run()
        assert len(scientific_downloads(fresh))==6
        widget=next(w for kind in ('number_input','selectbox','multiselect','checkbox') for w in fresh.get(kind) if w.key==control)
        widget.set_value(value).run()
        assert not fresh.exception and not fresh.get('download_button') and view.KEY not in fresh.session_state
    changed=d.copy();changed.iloc[0,1]=1-changed.iloc[0,1];at.session_state['raw_df']=changed
    at.run();assert not at.exception and not at.get('download_button') and view.KEY not in at.session_state


def test_public_dif_parity_anchor_sensitivity_and_suppressed_invalid_exports(binary,tmp_path):
    # Restrict to one item to verify the public API's add-design statistic once.
    from mirt.multigroup.invariance import invariance_lrt
    d=binary.groupby('Grup',sort=False).head(120).copy();items=list(d.columns[1:6]);d=d[['Grup']+items]
    args=dict(items=items,group='Grup',reference='R',focal='F',model='2PL',studied=[items[-1]],n_quadpts=21,tol=1e-5)
    first=analyze(d,anchors=items[:-1],**args)
    second=analyze(d,anchors=items[:2],**args)
    x,g,_,_,_=prepare(d,items,'Grup','R','F','2PL')
    native=mirt.multigroup_dif(x,g,model='2PL',items=[items[-1]],anchors=items[:-1],scheme='add',
            parameters=('intercepts',),n_quadpts=21,tol=1e-5,reference_group='reference')
    assert native.df.item()==first['tables']['DIF'].df.item()==1
    assert np.isfinite(native.chi2.item())
    # Optimizer starts differ: measure rather than invent a cross-optimizer tolerance.
    evidence=dict(public_multigroup_dif_LR=native.chi2.item(),retained_pair_LR=first['tables']['DIF'].LR.item(),
                  alternative_anchors_LR=second['tables']['DIF'].LR.item(),
                  status='MEASURED / NOT CERTIFIED',parameters='intercepts',n_quadpts=21,tol=1e-5)
    import os
    if os.environ.get('PSIKO_DIF_EVIDENCE'):
        path=Path(os.environ['PSIKO_DIF_EVIDENCE'])/'api_parity.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(evidence,indent=2)+'\n')
    # A real nonconverged library solution must retain diagnostics and suppress inference/downloads.
    failed=analyze(d,anchors=items[:2],**(args|dict(max_iter=1)))
    assert not failed['tables']['Fit'].Converged.all()
    assert failed['tables']['DIF'].Reason.str.contains('NOT CONVERGED').all()
    output=downloads(failed,items[-1])
    assert pd.read_csv(io.BytesIO(output['csv']))[['p','q']].isna().all().all()
    assert pd.read_excel(io.BytesIO(output['xlsx']),sheet_name='DIF')[['p','q']].isna().all().all()
    # Curve effect is exactly the defined maximum category probability contrast.
    diff=first['tables']['Difference'];eff=first['tables']['Effect']
    expected=diff.groupby('Item').Focal_minus_reference.apply(lambda x:float(np.abs(x).max()))
    np.testing.assert_array_equal(expected,eff.Max_abs_probability_difference_on_grid)
