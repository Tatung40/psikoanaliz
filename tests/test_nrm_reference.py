"""NRM-only no-fit contracts: nominal coding, reflection, scores and fixed budgets."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import mirt
from validation.nrm_r_comparison import marginal,orientation,acceptance


def test_nominal_probabilities_gradients_and_reflection():
    A=np.array([[0,.7,-.3,0],[0,-.8,.4,1.2]])
    D=np.array([[0,-.2,.4,.1],[0,.3,-.5,.2]])
    th=np.array([-2.,-1.,0.,1.,2.]);w=np.array([.05,.2,.5,.2,.05])
    x=np.array([[0,3],[2,1],[3,2],[1,0]])
    model=mirt.NominalResponseModel(2,4);model.set_parameters(slopes=A,intercepts=D)
    P=model.probability(th)
    ll,g=marginal(x,A,D,th,w)
    direct=sum(np.log(sum(w[q]*np.prod([P[q,i,row[i]] for i in range(2)]) for q in range(5))) for row in x)
    np.testing.assert_allclose(ll,direct,rtol=1e-13)
    np.testing.assert_allclose(marginal(x,-A,D,th,w)[0],ll,atol=1e-13)
    for key,arr in [('A',A),('D',D)]:
        for i in range(2):
            for c in range(1,4):
                ap,am,dp,dm=A.copy(),A.copy(),D.copy(),D.copy()
                up,dn=(ap,am) if key=='A' else (dp,dm);up[i,c]+=1e-5;dn[i,c]-=1e-5
                numerical=(marginal(x,ap,dp,th,w)[0]-marginal(x,am,dm,th,w)[0])/2e-5
                np.testing.assert_allclose(g[key][i,c],numerical,rtol=1e-6,atol=1e-8)
    assert orientation(A,A)==1 and orientation(-A,A)==-1
    with pytest.raises(ValueError,match='orientation'):orientation(np.zeros_like(A),A)


def test_precommitted_budgets_fail_instead_of_expanding():
    c=json.loads(Path('validation/nrm_acceptance.json').read_text())
    m={k:0. for k in c['max_absolute']};m.update(test_information=0.,aic=0.,bic=0.)
    assert all(v['passed'] for v in acceptance(m,0.,8,c).values())
    m['parameters']=c['max_absolute']['parameters']*1.01
    assert not acceptance(m,0.,8,c)['parameters']['passed']
    assert c['decision_if_all_pass']=='MEASURED / NOT CERTIFIED'


def test_nominal_diagnostics_do_not_call_ordinal_api(monkeypatch):
    from core.irt_b14_diagnostics import model_diagnostics
    monkeypatch.setattr(mirt,'itemfit',lambda *a,**k:pytest.fail('Ordinal itemfit forbidden for nominal categories'))
    status,tables=model_diagnostics(dict(model='NRM'))
    assert not tables and 'uydurulmaz' in status.Not.iloc[0]
