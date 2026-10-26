"""Native Python mirt 1.2.0 solver API forwarding and safety checks."""
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pytest
from core.irt_python_mirt_adapter import fit_python_mirt


def test_native_solver_settings_forwarded():
    x=np.tile([[0,1,0],[1,0,1],[1,1,0],[0,0,1]],(8,1))
    fake=SimpleNamespace()
    with patch('mirt.fit_mirt',return_value=fake) as fit:
        result=fit_python_mirt(x,model='Rasch',solver_settings={'max_iter':700,'tol':1e-6,'n_quadpts':41})
    assert result.fit is fake
    assert fit.call_args.kwargs['model']=='1PL'
    assert fit.call_args.kwargs['max_iter']==700
    assert fit.call_args.kwargs['tol']==1e-6
    assert fit.call_args.kwargs['n_quadpts']==41


@pytest.mark.parametrize('settings',[
    {'max_iter':0},{'max_iter':True},{'n_quadpts':2},{'tol':0},
    {'tol':float('nan')},{'not_real':4},[1,2,3],
])
def test_invalid_solver_settings_never_call_engine(settings):
    x=np.tile([[0,1,0],[1,0,1],[1,1,0],[0,0,1]],(8,1))
    with patch('mirt.fit_mirt') as fit:
        with pytest.raises(ValueError):
            fit_python_mirt(x,model='Rasch',solver_settings=settings)
        fit.assert_not_called()
