import json
import numpy as np
import pandas as pd
import pytest
from core import grm_dif as G
from validation.gh_reoptimization import experiment as E


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
