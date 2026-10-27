"""Actual general-CFA UI/calibration; not independent scientific evidence."""
import io,zipfile
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from core import irt_multidim_general_report as general


@pytest.mark.parametrize('variant',[0,1,2,3])
def test_actual_general_fit_id_posterior_exports_and_stale_inputs(monkeypatch,variant):
 import ui.irt_multidim_view as view
 data=pd.read_csv('validation/mirt_phase3/fixtures/2PL_2D.csv');items=list(data);data['ID']=[f'P{i}' for i in range(len(data))]
 calls=[];actual=general.fit_report
 def counted(*a,**kw):calls.append(kw);return actual(*a,**kw)
 monkeypatch.setattr(general,'fit_report',counted)
 at=AppTest.from_file(str(Path('app.py').resolve()),default_timeout=180)
 for k,v in dict(raw_df=data,scored_df=None,item_cols=items,id_col='ID',nav_page='Çok Boyutlu MTK (MIRT)').items():at.session_state[k]=v
 # Verified fixture has six F1 and four F2 items; simulate the actual editor
 # cell changes from its equal-sized default rather than changing observed data.
 at.session_state['md_pattern_10_2_False']={'edited_rows':{4:{'F1':0,'F2':1},8:{'F1':1,'F2':0},9:{'F1':1,'F2':0}},'added_rows':[],'deleted_rows':[]}
 at.run();at.selectbox(key='md_model').set_value(general.MODELS[variant]).run();assert not at.exception
 assert at.selectbox(key='md_general_quad_2').value==15
 at.button(key='md_run').click().run()
 assert not at.exception and not at.error,[e.value for e in at.error]
 assert len(calls)==1 and calls[0]['correlated']=='full' and calls[0]['tol']==1e-13
 result=at.session_state[view.KEY]['result'];meta=result['metadata'];tables=result['tables']
 assert meta['Converged'] and meta['Parameters']==[21,21,31,41][variant] and 'SciPy' in meta['Engine']
 assert tables['Pattern'].F1.tolist()==[1,1,1,1,0,0,0,0,1,1]
 assert abs(meta['LogLik']-[-2536.6369212147274,-2536.183910986355,-2527.1339764582062,-2526.762503677404][variant])<1e-6
 assert meta['MAP']=='NOT VALIDATED / DISABLED' and 'MAP' not in tables
 assert tables['EAP'].ID.tolist()==data.ID.tolist()
 assert len(tables['PosteriorCovariance'])==len(data)*4
 np.testing.assert_allclose(tables['Covariance'].iloc[:,1:].to_numpy().diagonal(),1,atol=1e-15)
 assert len(at.get('download_button'))==5
 bundle=at.session_state[view.KEY]['bundle']
 assert 'PosteriorCovariance' in pd.ExcelFile(io.BytesIO(bundle['xlsx'])).sheet_names
 with zipfile.ZipFile(io.BytesIO(bundle['zip'])) as archive:
  assert {'Parameters.csv','EAP.csv','OptimizerTrace.csv','Scores.csv'}.issubset(archive.namelist())
  note=archive.read('report.md').decode();assert 'NOT CERTIFIED' in note and 'SciPy' in note and 'no local estimator' not in note
 at.number_input(key='md_fixed_u' if variant==3 else 'md_fixed_c').set_value(.94 if variant==3 else .21).run()
 assert not at.exception and view.KEY not in at.session_state and not at.get('download_button') and len(calls)==1


def test_general_resource_and_unvalidated_modes_fail_before_fit(monkeypatch):
 import pytest
 data=pd.read_csv('validation/mirt_phase3/fixtures/2PL_2D.csv');p=np.zeros((10,2),int);p[:6,0]=1;p[6:,1]=1
 def forbidden(*a,**kw):raise AssertionError('Unsupported input reached calibration')
 monkeypatch.setattr(general.engine,'fit',forbidden)
 for kw in (dict(correlated=False),dict(exploratory=True)):
  with pytest.raises(ValueError):general.fit_report(data,general.MODELS[0],p,**kw)
 with pytest.raises(ValueError):general.fit_report(data,'4PL free unvalidated',p)
 cross=p.copy();cross[0,:]=1
 with pytest.raises(ValueError,match='2D between-item'):general.fit_report(data,general.MODELS[2],cross)
 wide=pd.concat([data.add_prefix(f'B{i}_') for i in range(23)],axis=1);pattern=np.repeat(np.eye(2),115,axis=0)
 with pytest.raises(ValueError,match='bellek'):general.fit_report(wide,general.MODELS[0],pattern)


def test_actual_nonconvergence_suppresses_scores_curves_and_png():
 from reports.irt_multidim_report import downloads
 data=pd.read_csv('validation/mirt_phase3/fixtures/2PL_2D.csv');p=np.zeros((10,2),int);p[:4,0]=1;p[4:8,1]=1;p[8:,0]=1
 result=general.fit_report(data,general.MODELS[0],p,n_quadpts=15,max_iter=1)
 assert not result['metadata']['Converged']
 assert not {'EAP','MAP','Curves','IIF','TIF','SEM'}.intersection(result['tables'])
 assert downloads(result,'M1','0')['png'] is None
