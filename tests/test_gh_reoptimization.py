import json
import numpy as np
import pandas as pd
import pytest
from core import grm_dif as G
from validation.gh_reoptimization import experiment as E
from validation.grm_archive_identity import bind_archived_models, assert_reconstructed_parameters


def test_versioned_protocol_and_saved_unique_constraints():
    protocol=json.loads((E.HERE/'protocol.json').read_text())
    assert protocol['version']=='GH-REOPT-1'
    assert protocol['stages']==[{'fit_points':101,'audit_points':161},{'fit_points':161,'audit_points':201}]
    assert E.protect()==152
    record,x,g,_=E.D.load_saved();models,slots=E.saved_models(record)
    assert len(models)==7 and len(slots)==8
    assert slots['P09.H2']==slots['P10.H2']
    for m in models.values():
        a,b,mu,sd=m['design'].unpack(m['params'])
        np.testing.assert_array_equal(a[0,:8],a[1,:8]);np.testing.assert_array_equal(b[0,:8],b[1,:8])
        lo,hi=np.array(m['design'].bounds).T
        assert np.all((m['params']>lo)&(m['params']<hi))


def test_native_api_grid_and_start_configuration_without_fit(monkeypatch):
    record,x,g,_=E.D.load_saved()
    inputs=G.validate_inputs(pd.DataFrame(x,columns=[f'P{i:02d}' for i in range(1,11)]),g,0,1,[f'P{i:02d}' for i in range(1,9)],categories=4)
    fitter=G.Fitter(inputs,points=101);fitter.audit_grid=E.D.reference_grid(161)
    assert len(fitter.grid[0])==101 and len(fitter.audit_grid[0])==161
    assert np.exp(fitter.audit_grid[1]).sum()==pytest.approx(1.,abs=1e-13)
    def forbidden(*a,**kw):raise AssertionError('No calibration in tests/CI')
    monkeypatch.setattr(G,'minimize',forbidden)
    models,_=E.saved_models(record)
    assert all(E.anchor_feasible(m) for m in models.values())


def test_native_lr_table_and_derived_error_bound_without_fit():
    slots={};solutions={}
    for item in ('P09','P10'):
        for hypothesis,ll in (('H0',-100.),('Hb',-98.),('Ha',-97.),('H2',-96.)):
            name=item+'.'+hypothesis;slots[name]=name
            solutions[name]={'evaluations':{'101':{'loglik':ll},'161':{'loglik':ll+.0001}}}
    table=E.lr_table(slots,solutions,101,161)
    assert len(table)==6
    assert all(r['valid'] and r['audit_minus_fit_lr']==pytest.approx(0.) for r in table)
    assert [r['df'] for r in table[:3]]==[4,3,1]
    assert all(r['implied_grid_error_bound']==.004 for r in table)


def test_cached_reoptimization_and_dense_gates_without_calibration():
    r=json.loads((E.HERE/'results/report.json').read_text())
    assert E.protect()==r['prior_files_verified']==152
    assert r['p_values_generated'] is False and r['previous_mc_unchanged']
    assert r['scientific_classification']=='SINGLE DATASET / NOT CERTIFIED'
    for path,digest in r['sources'].items():assert E.D.sha((E.ROOT/path).read_bytes())==digest
    prior,x,g,path=E.D.load_saved();models,slots=E.saved_models(prior)
    assert r['data_sha256']==prior['data_sha256']
    models,slots=bind_archived_models(r,models,slots)
    assert E.D.sha(path.read_bytes())==r['source_archive_sha256']
    assert all(v['p'] is None for v in prior['result']['table'])
    assert [s['fit_points'] for s in r['stages']] in ([101],[101,161])
    if len(r['stages'])==2:assert not r['stages'][0]['passed']
    for stage in r['stages']:
        assert stage['audit_points']>stage['fit_points']
        if stage['passed']:assert len(stage['solutions'])==stage['native_optimizer_calls']==7
        for key,s in stage['solutions'].items():
            if stage['fit_points']==101:
                assert_reconstructed_parameters(s['initial_params'],models[key]['params'],models[key]['design'])
            a,b=np.array(s['a']),np.array(s['b'])
            np.testing.assert_array_equal(a[0,:8],a[1,:8]);np.testing.assert_array_equal(b[0,:8],b[1,:8])
            assert (np.diff(b,axis=2)>0).all()
            audit=s['evaluations'][str(stage['audit_points'])]
            if s['passed']:
                assert s['converged'] and not s['at_bound'] and s['covariance_available']
                assert s['native_score']<2e-6 and audit['max_abs_normalized_gradient']<=1e-5
                assert s['quadrature_error']<=1e-3 and s['quadrature_ok']
                eigen=np.array(s['dense_information']['eigenvalues'])
                assert eigen[0]>1e-7*max(eigen[-1],1.)
            for q,e in s['evaluations'].items():
                assert e['independent_loglik']==pytest.approx(e['loglik'],abs=1e-8)
                assert np.max(np.abs(e['gradient']))==e['max_abs_normalized_gradient']
                assert e['second_moment']==pytest.approx(1.,abs=1e-13)
            for h,fd in s['derivative_checks'].items():
                if h=='reason':continue
                error=np.max(np.abs(np.array(fd['finite_difference_gradient'])-audit['gradient']))
                assert fd['max_abs_error']==error
        for row in stage['lr']:
            a=stage['solutions'][slots[row['item']+'.H2']]
            null={'omnibus':'H0','thresholds':'Ha','slope':'Hb'}[row['test']]
            b=stage['solutions'][slots[row['item']+'.'+null]]
            for q,stat in row['lr'].items():
                expected,_=G._nested_lr(a['evaluations'][q]['loglik'],b['evaluations'][q]['loglik'])
                assert stat==expected
    model=next(iter(models.values()));d=model['design'];p=model['params'].copy()
    j=next(j for j,label in enumerate(d.labels) if label[0]=='b' and label[3]>0)
    changed=p.copy();changed[j]=np.nextafter(np.nextafter(changed[j],np.inf),np.inf)
    with pytest.raises(AssertionError):assert_reconstructed_parameters(p,changed,d)
    changed=p.copy();changed[0]=np.nextafter(changed[0],np.inf)
    with pytest.raises(AssertionError):assert_reconstructed_parameters(p,changed,d)
    assert r['fits_attempted']>=r['fits_executed']
