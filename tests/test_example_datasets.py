"""Real loader, scoring and GeneralizIT; no estimator certification from schemas."""
import ast
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import example_datasets as E
from core.answer_keys import validate_key,read_key
from core.data_processing import load_dataset

ROOT=Path(__file__).resolve().parents[1]


def test_registry_covers_actual_routes_and_has_three_distinct_levels():
    routed=set()
    for n in ast.walk(ast.parse((ROOT/'app.py').read_text())):
        if isinstance(n,ast.Compare) and isinstance(n.left,ast.Name) and n.left.id=='selected_page':
            for c in n.comparators:
                if isinstance(c,ast.Constant) and isinstance(c.value,str):routed.add(c.value)
                elif isinstance(c,(ast.Tuple,ast.List)):routed.update(v.value for v in c.elts if isinstance(v,ast.Constant))
    excluded={'1. Ana Sayfa','2. Veri Yükleme','11. Formüller ve Yöntemler','12. Kullanım Kılavuzu'}
    assert routed-excluded <= set(E.analysis_catalog())
    assert all(len(E.available(page))>=3 for page in E.analysis_catalog())
    r=list(E.registry().values())
    for family in ('irt','mirt','g'):
        for model in {d.model for d in r if d.family==family}:
            group=[d for d in r if d.family==family and d.model==model]
            assert sorted(d.level for d in group)==[0,1,2]
            assert len({(d.n,d.k,d.seed) for d in group})==3
    assert len([d for d in r if d.family=='g'])==15
    assert not E.available('unsupported') and not E.available('11. Formüller ve Yöntemler')


def test_response_and_latent_rng_streams_are_distinct(monkeypatch):
    import mirt
    actual=mirt.simdata;seeds=[]
    def record(*args,**kwargs):
        seeds.append(kwargs['seed'])
        return actual(*args,**kwargs)
    monkeypatch.setattr(mirt,'simdata',record)
    d=next(d for d in E.registry().values() if d.model=='2PL' and d.family=='irt')
    E.materialize(d.id)
    assert seeds==[int(np.random.SeedSequence([d.seed,104729]).generate_state(1)[0])]
    assert seeds[0]!=d.seed


@pytest.mark.parametrize('dataset_id',list(E.registry()))
def test_deterministic_schema_real_loader_and_no_shared_mutation(dataset_id):
    first=E.materialize(dataset_id);other=E.materialize(dataset_id)
    pd.testing.assert_frame_equal(first['data'],other['data'])
    d=first['spec'];frame=first['data'];items=first['items']
    assert len(frame)==d.n and frame.columns.is_unique
    assert all(c in frame for c in items)
    assert first['metadata']['synthetic'] and first['metadata']['seed']==d.seed
    if d.family=='mirt':
        assert np.linalg.matrix_rank(np.asarray(first['metadata']['loadings']))==d.dimensions
    actual=E.input_frame(first)
    pd.testing.assert_frame_equal(actual,frame,check_dtype=False,rtol=1e-14,atol=1e-14)
    if d.family!='g':assert frame[first['id']].is_unique and frame[first['group']].nunique()==2
    if d.family not in ('missing',):assert not frame.isna().any().any()
    first['data'].loc[0,first['id']]='MUTATED'
    assert E.materialize(dataset_id)['data'].loc[0,first['id']]!='MUTATED'


@pytest.mark.parametrize('dataset_id',[d.id for d in E.registry().values() if d.family=='g'])
def test_all_fifteen_real_generalizit_g_and_d(dataset_id,record_property):
    from core.generalizability import prepare,analyze
    b=E.materialize(dataset_id);m=b['metadata'];d=b['spec']
    _,counts=prepare(b['data'],d.design,m['columns'],'score')
    assert counts==m['counts']
    if d.design=='p x (r:i)':
        effects=np.asarray(m['realized_effects']['r'])
        assert effects.shape==(counts['i'],counts['r'])
        assert not np.array_equal(effects[0],effects[1])
    if d.design=='p x (i:r)':
        effects=np.asarray(m['realized_effects']['i'])
        assert effects.shape==(counts['r'],counts['i'])
        assert not np.array_equal(effects[0],effects[1])
    out=analyze(b['data'],d.design,m['columns'],'score',m['scenarios'])
    assert out['g'].shape==(1,2) and len(out['d']) in (2,4)
    assert np.isfinite(out['g'].to_numpy()).all()
    assert (out['d'][['rho^2','phi^2']].to_numpy()<=1).all()
    assert (out['d'][['rho^2','phi^2']].to_numpy()>=0).all()
    assert out['d']['universe'].gt(0).all()
    record_property(dataset_id,json.dumps(out['g'].to_dict()))


@pytest.mark.parametrize('dataset_id',[d.id for d in E.registry().values() if d.family=='abc'])
def test_abc_all_formats_keys_and_real_ctt(dataset_id):
    from core.test_statistics import compute_test_summary
    from core.reliability import compute_reliability_indices
    b=E.materialize(dataset_id);items=b['items'];raw=b['data']
    assert set(raw[items].to_numpy().ravel())==set('ABCDE')
    key=validate_key(b['key'],items,raw)
    scored=E.scored(b)
    np.testing.assert_array_equal(scored[items],raw[items].eq(pd.Series(key)).astype(float))
    if len(items)==4:np.testing.assert_array_equal(scored[items].iloc[:3],[[1,1,1,1],[0,1,0,0],[1,0,1,1]])
    summary=compute_test_summary(scored[items].sum(axis=1),len(items))
    reliability=compute_reliability_indices(scored,items)
    assert isinstance(summary,dict) and isinstance(reliability,dict)
    files=E.downloads(b)
    for ext in ('csv','xlsx','tsv','txt'):
        fmt='csv' if ext in ('tsv','txt') else ext
        frame,_=load_dataset(files['veri.'+ext],file_type=fmt,delimiter='\t' if ext in ('txt','tsv') else ',')
        pd.testing.assert_frame_equal(frame,raw)
    for ext in ('csv','xlsx'):
        assert read_key(files['cevap_anahtari.'+ext],ext,items,raw)==key
    book=pd.ExcelFile(io.BytesIO(files['tum_sayfalar.xlsx']))
    assert book.sheet_names==['veri','cevap_anahtari','puanlanmis']
    assert read_key(files['tum_sayfalar.xlsx'],'xlsx',items,raw)==key
    if len(items) in (20,50):assert len(scored[items].columns)==len(items)


@pytest.mark.parametrize('fault',['missing','extra','invalid','duplicate','numeric'])
def test_bad_keys_fail_closed(fault):
    b=next(E.materialize(d.id) for d in E.registry().values() if d.family=='abc')
    key=dict(b['key']);items=b['items'];raw=b['data'].copy()
    if fault=='missing':key.pop(items[0])
    if fault=='extra':key['NOT_AN_ITEM']='A'
    if fault=='invalid':key[items[0]]='AA'
    if fault=='numeric':raw[items[0]]=1
    if fault=='duplicate':
        content='madde,dogru_cevap\nM01,A\nM01,B\n'.encode()
        with pytest.raises(ValueError,match='yinelenen'):read_key(content,'csv',items,raw)
    else:
        with pytest.raises(ValueError):validate_key(key,items,raw)


def test_duplicate_key_headers_and_duplicate_requested_items_rejected():
    with pytest.raises(ValueError,match='başlık'):
        read_key(b'madde,madde,dogru_cevap\nM01,M02,A\n','csv',['M01'])
    with pytest.raises(ValueError,match='benzersiz'):
        validate_key({'M01':'A'},['M01','M01'])


@pytest.mark.parametrize('model',['2PL','GRM','PCM','GPCM','NRM'])
def test_real_library_calibration_of_basic_model_data(model,tmp_path):
    from core.irt_b14 import prepare_polytomous
    from core.irt_python_mirt_adapter import fit_python_mirt
    d=next(d for d in E.registry().values() if d.family=='irt' and d.model==model and d.level==0)
    b=E.materialize(d.id);x=b['data'][b['items']]
    if model!='2PL':x,_=prepare_polytomous(x,model,categories=list('ABCD') if model=='NRM' else None)
    fit=fit_python_mirt(x.to_numpy(),model=model,n_categories=None if model=='2PL' else 4).fit
    (tmp_path/'actual_calibration.json').write_text(json.dumps({'dataset':d.id,'model':model,
        'converged':bool(fit.converged),'loglik':float(fit.log_likelihood),'certified':False}))
    assert fit.converged, 'Actual example calibration failed; do not claim engine validation'
    assert np.isfinite(fit.log_likelihood)


def test_guide_and_registry_are_current_and_nominal_not_ordinal():
    from ui.example_datasets_view import guide
    assert callable(guide)
    assert not any(d.model=='RSM' for d in E.available('MTK: Python mirt NRM'))
    from core.python_mirt_dif import MODELS
    assert all(d.model in MODELS for d in E.available('MTK: Python mirt DIF'))
    for page in ('Yerel Bağımsızlık ve Madde Uyumu','MTK: Karşılaştırma'):
        assert all(d.model in ('ABCDE','2PL') for d in E.available(page))
    for d in E.registry().values():
        if d.family=='dif' and d.model in ('PCM','RSM'):assert d.effect!='nonuniform'
