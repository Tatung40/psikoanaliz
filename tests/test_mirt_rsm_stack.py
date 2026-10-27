"""Exact shared-step likelihood identity, fail-closed public API and actual UI."""
import io,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from mirt.models.polytomous import RatingScaleModel
from mirt.estimation.quadrature import GaussHermiteQuadrature
from core import irt_multidim_rsm_stack as engine
from validation.mirt_stage1.compare import likelihood


def data_pattern(q=2,n=120):
 raw=pd.read_csv('validation/evidence/p5_browser/poly_ordinal.csv');parts=[]
 for d in range(q):parts.append(raw.iloc[d*n:(d+1)*n,:6].reset_index(drop=True).add_prefix(f'F{d+1}_'))
 data=pd.concat(parts,axis=1);p=np.zeros((6*q,q),int)
 for d in range(q):p[d*6:(d+1)*6,d]=1
 return data,p


@pytest.mark.parametrize('q',[2,3,4,5])
def test_shared_global_steps_exact_stack_likelihood_and_information(q):
 data,p=data_pattern(q,4);m=RatingScaleModel(len(p),4)
 # Disordered Andrich steps are allowed; these are not GRM thresholds.
 m.set_parameters(difficulty=np.linspace(-.6,.6,len(p)),thresholds=np.array([0.,.7,-.2]))
 curves=engine.Curves(m,p,4);grid=GaussHermiteQuadrature(5,q);one=GaussHermiteQuadrature(5,1)
 stacked=engine.stack_responses(data,p)
 native=float(logsumexp(m.log_likelihood_batch(stacked,one.nodes)+np.log(one.weights),axis=1).sum())
 joint=likelihood(data.to_numpy(),curves.probability(grid.nodes),grid.weights)
 np.testing.assert_allclose(native,joint,rtol=0,atol=1e-12)
 assert np.count_nonzero(stacked>=0)==data.size
 assert m.n_parameters==len(p)+2 and m.parameters['thresholds'][0]==0
 theta=np.array([[0.]*q,[.5]*q]);I=curves.test_information_matrix(theta)
 np.testing.assert_allclose(I,sum(curves.item_information_matrix(theta,j) for j in range(len(p))),atol=1e-12)
 for d in range(q):np.testing.assert_allclose(I[:,d,d],sum(m.information(theta[:,d,None],item_idx=int(j)) for j in np.flatnonzero(p[:,d])))
 off=I.copy();off[:,np.arange(q),np.arange(q)]=0;assert not off.any()
 # Native item information equals category-probability derivative information.
 h=1e-5;hi=curves.probability(theta+h);lo=curves.probability(theta-h);P=curves.probability(theta)
 manual=(((hi-lo)/(2*h))**2/P).sum(2)
 for j in range(len(p)):np.testing.assert_allclose(curves.item_information_matrix(theta,j)[:,np.flatnonzero(p[j])[0],np.flatnonzero(p[j])[0]],manual[:,j],rtol=1e-7,atol=1e-8)


def test_rsm_rejects_wrong_structure_missing_and_budget_before_fit(monkeypatch):
 data,p=data_pattern();calls=[]
 def forbidden(*a,**kw):calls.append(1);raise AssertionError('Invalid input reached estimator')
 monkeypatch.setattr(engine.mirt,'EMEstimator',forbidden)
 for options in (dict(correlated=True),dict(exploratory=True),dict(n_quadpts=100)):
  with pytest.raises(ValueError):engine.fit(data,engine.NAME,p,**options)
 cross=p.copy();cross[0,:]=1
 with pytest.raises(ValueError):engine.fit(data,engine.NAME,cross)
 broken=data.copy();broken.iloc[0,0]=np.nan
 with pytest.raises(ValueError):engine.fit(broken,engine.NAME,p)
 assert not calls


def test_actual_rsm_ui_global_steps_scores_exports_and_stale_labels():
 from streamlit.testing.v1 import AppTest
 import ui.irt_multidim_view as view
 data,p=data_pattern();data=data+1;items=list(data);data['ID']=[f'R{i}' for i in range(len(data))]
 at=AppTest.from_file(str(Path('app.py').resolve()),default_timeout=180)
 for k,v in dict(raw_df=data,scored_df=None,item_cols=items,id_col='ID',nav_page='Çok Boyutlu MTK (MIRT)').items():at.session_state[k]=v
 at.run();at.selectbox(key='md_model').set_value(engine.NAME).run();assert not at.exception
 at.button(key='md_run').click().run();assert not at.exception and not at.error, [e.value for e in at.error]
 result=at.session_state[view.KEY]['result'];meta=result['metadata'];tables=result['tables']
 assert meta['Converged'],meta
 assert meta['Parameters']==14 and meta['N']==120 and meta['Native_stacked_N']==240
 assert meta['Observed_responses']==data[items].size and meta['Labels']==('1','2','3','4')
 assert len(tables['SharedSteps'])==3 and tables['SharedSteps'].Tau.iloc[0]==0
 assert len(tables['Fit'])==1 and tables['Parameters'].SharedStepGroup.nunique()==1
 assert tables['EAP'].ID.tolist()==data.ID.tolist() and tables['MAP'].ID.tolist()==data.ID.tolist()
 np.testing.assert_allclose(meta['BIC'],-2*meta['LogLik']+14*np.log(120))
 bundle=at.session_state[view.KEY]['bundle'];assert len(at.get('download_button'))==5
 with zipfile.ZipFile(io.BytesIO(bundle['zip'])) as z:
  assert {'SharedSteps.csv','StructuralRows.csv','EAP.csv','MAP.csv'}.issubset(z.namelist())
  assert 'NOT CERTIFIED' in z.read('report.md').decode()
 assert pd.ExcelFile(io.BytesIO(bundle['xlsx'])).sheet_names.count('SharedSteps')==1
 # Identical shape and columns with changed actual category labels is stale.
 at.session_state['raw_df']=data.assign(**{items[0]:data[items[0]].map({1:2,2:1,3:3,4:4})});at.run()
 assert not at.exception and not at.get('download_button') and view.KEY not in at.session_state


def test_nonconvergence_closes_scores_curves_and_png():
 from types import SimpleNamespace
 from reports.irt_multidim_report import downloads
 data,p=data_pattern();m=RatingScaleModel(12,4)
 fake=SimpleNamespace(model=m,n_parameters=14,converged=False,n_iterations=1,log_likelihood=-100.,aic=228.,bic=280.)
 result=engine.report(dict(fit=fake,coded=data,pattern=p.astype(bool),labels=[0,1,2,3],settings=dict(n_quadpts=21),stacked_rows=240,model=engine.NAME))
 assert not {'EAP','MAP','Curves','IIF','TIF','SEM'}.intersection(result['tables'])
 assert downloads(result,'F1_I1',0)['png'] is None


def test_rsm_no_r_means_blocked_without_python_fit(tmp_path,monkeypatch):
 from validation.mirt_phase2 import rsm_comparison as harness
 monkeypatch.setattr(harness.shutil,'which',lambda _:None)
 def forbidden(*a,**kw):raise AssertionError('Python fit before independent R preflight')
 monkeypatch.setattr(harness.engine,'fit',forbidden)
 result=harness.execute_case(2,200,tmp_path/'rsm')
 assert result['status']=='BLOCKED' and result['R_execution']=='NOT EXECUTED'
 assert result['Python_execution']=='NOT EXECUTED' and 'Rscript unavailable' in result['blocker']
 assert (tmp_path/'rsm/data.csv').exists() and (tmp_path/'rsm/stacked.csv').exists()
 # Mapping contract: native genuine rsm, explicit equality ties, zero-step gauge.
 source=Path('validation/mirt_phase2/rsm_reference.R').read_text()
 assert "itemtype='rsm'" in source and 'internal_constraints=FALSE' in source
 assert "'CONSTRAIN = '" in source and "sv$est[index]<-k!=1" in source
 assert 'identical(attr(sv,k),attrs[[k]])' in source


def test_rsm_native_rasch_variance_is_fixed_before_both_r_paths():
 source=Path('validation/mirt_phase2/rsm_reference.R').read_text()
 # Native rsm defaults to estimating variance (actual run 40470531009).
 # Guard the intended fixed latent scale, not a posthoc variance rescaling.
 fixed="sv$value[sv$name=='COV_11']<-1"
 flags="sv$est[grepl('^(MEAN|COV)_',sv$name)]<-FALSE"
 assert fixed in source and flags in source
 assert source.index(fixed)<source.index('fit<-do.call(mirt,base)')
 assert source.index(flags)<source.index("if(mode=='preflight')sv$est[]<-FALSE")
 assert "file.path(out,'raw_parameters.csv')" in source.split('stopifnot(all(values$value')[0]


def test_public_fixed_evaluation_restores_scoring_without_updating_parameters():
 import mirt
 from validation.mirt_phase2.rsm_comparison import restore_saved_python
 data,p=data_pattern(2,30);model=RatingScaleModel(12,4);values=model.parameters
 grid=GaussHermiteQuadrature(61,1);stacked=engine.stack_responses(data,p)
 ll=float(logsumexp(model.log_likelihood_batch(stacked,grid.nodes)+np.log(grid.weights),axis=1).sum())
 # Container restoration contract only; this mock metadata is NOT evidence
 # that these fixed parameters were calibrated or scientifically accepted.
 metadata=dict(logLik=ll,iterations=123,converged=True,AIC=-2*ll+28,Native_stacked_BIC=-2*ll+14*np.log(60),free_parameters=14)
 restored=restore_saved_python(values,metadata,data,p)
 assert restored.n_iterations==123 and restored.n_parameters==14 and restored.model.is_fitted
 for key in values:np.testing.assert_array_equal(restored.model.parameters[key],values[key])
 result=mirt.fscores(restored,data.iloc[:2].to_numpy(),method='EAP',n_quadpts=61)
 assert np.isfinite(result.theta).all() and np.isfinite(result.standard_error).all()
