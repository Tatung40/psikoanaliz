"""Observe actual public library EM iterations, stopping and GH grids, not fake fits."""
import inspect,json
import numpy as np
import pandas as pd
import pytest
import mirt
from core.irt_python_mirt_adapter import fit_python_mirt,solver_settings
from core.irt_binary_unified import fit_binary


def binary_data():
    return pd.read_csv('tests/fixtures/rasch/data_rasch_main.csv').iloc[:120,:6].astype(int)


def test_real_em_settings_and_unchanged_defaults(monkeypatch,record_property):
    from mirt.estimation.em import EMEstimator
    actual=[];original=EMEstimator.fit
    def observe(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        actual.append(dict(tol=self.tol,max_iter=self.max_iter,nodes=self._quadrature.nodes.copy(),
            weights=self._quadrature.weights.copy(),history=self.convergence_history))
        return result
    monkeypatch.setattr(EMEstimator,'fit',observe)
    x=binary_data()
    one=fit_python_mirt(x,model='1PL',max_iter=1,tol=1e-12,n_quadpts=11)
    three=fit_binary(x,'1PL',engine='python_mirt',max_iter=3,tol=1e-12,n_quadpts=31).result
    loose=fit_python_mirt(x,model='1PL',max_iter=20,tol=1e6,n_quadpts=31)
    strict=fit_python_mirt(x,model='1PL',max_iter=20,tol=1e-8,n_quadpts=31)
    assert one.fit.n_iterations==1 and not one.fit.converged
    assert three.fit.n_iterations==3 and not three.fit.converged
    assert loose.fit.converged and loose.fit.n_iterations==2
    assert strict.fit.n_iterations>loose.fit.n_iterations
    for capture,expected in zip(actual,[(1,1e-12,11),(3,1e-12,31),(20,1e6,31),(20,1e-8,31)]):
        maximum,tol,grid=expected
        assert (capture['max_iter'],capture['tol'],len(capture['nodes']))==expected
        assert len(capture['history'])<=maximum+1  # Library evaluates final LL once after capped M-steps.
        np.testing.assert_allclose(capture['weights'].sum(),1,atol=1e-14)
        np.testing.assert_allclose(capture['weights']@capture['nodes'][:,0]**2,1,atol=1e-14)
    assert abs(actual[2]['history'][-1]-actual[2]['history'][-2])<actual[2]['tol']
    if strict.fit.converged:assert abs(actual[3]['history'][-1]-actual[3]['history'][-2])<actual[3]['tol']
    record_property('actual_solver',json.dumps(dict(iterations=[one.fit.n_iterations,three.fit.n_iterations,loose.fit.n_iterations,strict.fit.n_iterations],
        settings=[dict(max_iter=a['max_iter'],tol=a['tol'],nodes=a['nodes'].tolist(),weights=a['weights'].tolist(),history=a['history']) for a in actual])))
    defaults=solver_settings()
    assert defaults=={k:inspect.signature(mirt.fit_mirt).parameters[k].default for k in defaults}
    a=fit_python_mirt(x,model='1PL')
    b=mirt.fit_mirt(x.to_numpy(),model='1PL')
    assert a.fit.log_likelihood==b.log_likelihood
    pd.testing.assert_frame_equal(a.fit.coef(),b.coef())
    assert a.fit.n_iterations==b.n_iterations and a.fit.converged==b.converged


@pytest.mark.parametrize('kwargs',[
    {'max_iter':0},{'max_iter':1.2},{'max_iter':True},{'max_iter':float('inf')},
    {'n_quadpts':0},{'n_quadpts':3.5},{'n_quadpts':False},
    {'tol':0},{'tol':-1},{'tol':float('nan')},{'tol':float('inf')},{'tol':True},{'tol':'1e-4'}])
def test_invalid_settings_fail_before_engine(kwargs,monkeypatch):
    import core.irt_python_mirt_adapter as adapter
    monkeypatch.setattr(adapter,'import_module',lambda *a:pytest.fail('Invalid setting reached mirt'))
    with pytest.raises(ValueError):fit_python_mirt(binary_data(),model='1PL',**kwargs)


def test_unknown_service_option_is_not_ignored():
    with pytest.raises(ValueError,match='Unsupported'):
        fit_binary(binary_data(),'2PL',engine='python_mirt',points=31)


def test_legacy_ui_applies_real_cap_and_invalidates_settings():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    data=binary_data()
    at=AppTest.from_file(str(Path('app.py').resolve()),default_timeout=120)
    for k,v in dict(raw_df=data,scored_df=data.copy(),item_cols=list(data.columns),id_col=None,group_col=None,is_dichotomous=True,nav_page='MTK: 1PL').items():at.session_state[k]=v
    at.run()
    prefix='py_mirt_solver_1PL'
    at.number_input(key=prefix+'_max_iter').set_value(1)
    at.number_input(key=prefix+'_tol').set_value(1e-12)
    at.number_input(key=prefix+'_n_quadpts').set_value(31).run()
    at.button(key='py_mirt_run_1PL').click().run()
    assert not at.exception
    out=at.session_state['python_mirt_experimental_results']['1PL']['fit']
    assert out.solver_settings==dict(max_iter=1,tol=1e-12,n_quadpts=31)
    assert out.fit.n_iterations==1 and not out.fit.converged
    assert any(e.value.startswith('Motor: python_mirt | Model:') for e in at.caption)
    at.number_input(key=prefix+'_n_quadpts').set_value(41).run()
    assert not at.exception
    assert not any(e.value.startswith('Motor: python_mirt | Model:') for e in at.caption)  # old result panel not rendered.
