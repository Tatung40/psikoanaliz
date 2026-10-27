"""Public-library factorization mathematics and real Streamlit route."""
import io,itertools,zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from mirt.models.dichotomous import ThreeParameterLogistic
from mirt.estimation.quadrature import GaussHermiteQuadrature
from core import irt_multidim_factorized as module
from validation.mirt_stage1.compare import likelihood


@pytest.mark.parametrize('q',[2,3,4,5])
def test_exact_factorization_native_probability_and_full_information(q):
 raw=pd.read_csv('tests/fixtures/irt_b11/data_irt3pl_ok.csv').iloc[:4,:6].to_numpy(dtype=int);data=np.tile(raw,(1,q))
 pattern=np.zeros((6*q,q),int);blocks=[]
 for d in range(q):
  index=np.arange(6*d,6*(d+1));pattern[index,d]=1
  m=ThreeParameterLogistic(6);m.set_parameters(discrimination=np.linspace(.7,1.4,6),difficulty=np.linspace(-.5,.5,6),guessing=np.full(6,.2))
  blocks.append((d,index,SimpleNamespace(model=m)))
 curves=module.Curves(blocks,pattern);full=GaussHermiteQuadrature(5,q);one=GaussHermiteQuadrature(5,1)
 joint=likelihood(data,curves.probability(full.nodes),full.weights)
 separate=sum(likelihood(data[:,idx],f.model.probability(one.nodes),one.weights) for _,idx,f in blocks)
 np.testing.assert_allclose(joint,separate,atol=1e-12)
 theta=np.array([[0.]*q,[.5]*q]);I=curves.test_information_matrix(theta)
 np.testing.assert_allclose(I,sum(curves.item_information_matrix(theta,j) for j in range(6*q)),atol=1e-12)
 for d in range(q):np.testing.assert_allclose(I[:,d,d],blocks[d][2].model.information(theta[:,d,None]).sum(1))
 off=I.copy();off[:,np.arange(q),np.arange(q)]=0;assert not off.any()


def test_restricted_structure_and_asymptote_gates_before_fit(monkeypatch):
 data=pd.read_csv('tests/fixtures/irt_b11/data_irt3pl_ok.csv').iloc[:60,:12]
 p=np.zeros((12,2),int);p[:6,0]=1;p[6:,1]=1
 def forbidden(*a,**kw):raise AssertionError('Invalid input reached estimator')
 monkeypatch.setattr(module.mirt,'fit_mirt',forbidden)
 for kwargs in (dict(correlated=True),dict(exploratory=True),dict(c=-.1),dict(c=.5,u=.5),dict(u=1.1)):
  with pytest.raises(ValueError):module.fit(data,module.MODELS[0],p,**kwargs)
 cross=p.copy();cross[0,:]=1
 with pytest.raises(ValueError,match='between-item'):module.fit(data,module.MODELS[0],cross)


@pytest.mark.parametrize('family',['3PL','4PL'])
def test_actual_factorized_ui_scores_exports_and_stale_asymptote(monkeypatch,family):
 from streamlit.testing.v1 import AppTest
 from reports.irt_multidim_report import downloads
 import ui.irt_multidim_view as view
 data=pd.read_csv(f'tests/fixtures/irt_b11/data_irt{family.lower()}_ok.csv').iloc[:120,:12].copy();items=list(data)
 data['ID']=[f'C{i}' for i in range(len(data))];calls=[];actual=module.fit
 def counted(*a,**kw):calls.append(kw);return actual(*a,**kw)
 monkeypatch.setattr(module,'fit',counted)
 at=AppTest.from_file(str(Path('app.py').resolve()),default_timeout=180)
 for k,v in dict(raw_df=data,scored_df=None,item_cols=items,id_col='ID',nav_page='Çok Boyutlu MTK (MIRT)').items():at.session_state[k]=v
 at.run();at.selectbox(key='md_model').set_value(next(m for m in module.MODELS if m.startswith(family))).run();assert not at.exception
 at.button(key='md_run').click().run();assert not at.exception and len(calls)==1
 result=at.session_state[view.KEY]['result'];assert result['metadata']['Converged'],result['metadata']
 assert (result['tables']['Parameters'].c==.2).all()
 assert (result['tables']['Parameters'].d==(.95 if family=='4PL' else 1.)).all()
 assert (result['tables']['Fit'].Free_parameters==12).all()
 assert result['tables']['EAP'].ID.tolist()==data.ID.tolist()
 np.testing.assert_allclose(result['metadata']['LogLik'],result['tables']['Fit'].LogLik.sum())
 assert len(at.get('download_button'))==5
 bundle=at.session_state[view.KEY]['bundle']
 with zipfile.ZipFile(io.BytesIO(bundle['zip'])) as z:
  assert 'Fit.csv' in z.namelist() and 'EAP.csv' in z.namelist()
  assert 'NOT CERTIFIED' in z.read('report.md').decode()
 at.number_input(key='md_fixed_u' if family=='4PL' else 'md_fixed_c').set_value(.94 if family=='4PL' else .21).run();assert not at.exception and not at.get('download_button') and view.KEY not in at.session_state
 assert len(calls)==1
