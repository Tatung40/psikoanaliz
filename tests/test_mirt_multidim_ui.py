"""Actual library flow once; reuse that result for stale-input checks."""
import io,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from core.irt_multidim import fit,report
from reports.irt_multidim_report import downloads
from validation.mirt_stage1.compare import inputs


def test_mirt_actual_ui_and_stale_downloads(tmp_path,monkeypatch):
 import ui.irt_multidim_view as view
 cfg,data,_,_=inputs('2PL_2D',tmp_path/'data');data=data.head(120).copy()
 data.insert(0,'ID',[f'P{i}' for i in range(len(data))]);items=list(data.columns[1:])
 calls=[];actual=view.fit
 def observed(*a,**kw):calls.append(kw);return actual(*a,**kw)
 monkeypatch.setattr(view,'fit',observed)
 def page(d=data):
  at=AppTest.from_file(str(Path('app.py').resolve()),default_timeout=180)
  for k,v in dict(raw_df=d,scored_df=None,item_cols=items,id_col='ID',nav_page='Çok Boyutlu MTK (MIRT)').items():at.session_state[k]=v
  at.run();assert not at.exception
  return at
 at=page();at.button(key='md_run').click().run();assert not at.exception
 assert len(calls)==1
 saved=at.session_state[view.KEY];result=saved['result']
 assert result['metadata']['Converged'],result['metadata']
 assert len(at.get('download_button'))==5
 assert result['tables']['EAP'].ID.tolist()==data.ID.tolist()
 assert np.isfinite(result['tables']['MAP'].filter(regex='^F').to_numpy()).all()
 assert not any('SE_' in c for c in result['tables']['MAP'])
 bundle=saved['bundle'];book=pd.ExcelFile(io.BytesIO(bundle['xlsx']))
 assert {'Parameters','Boundaries','EAP','MAP','TIF'}<=set(book.sheet_names)
 with zipfile.ZipFile(io.BytesIO(bundle['zip'])) as z:
  assert {'metadata.json','results.xlsx','report.md','surfaces.png','TIF.csv'}<=set(z.namelist())
  assert 'NOT CERTIFIED' in z.read('report.md').decode()
 at.run();assert len(calls)==1 and len(at.get('download_button'))==5
 for control,value in [('md_model','GRM'),('md_dims',3),('md_corr','Tümü'),('md_iter',100),('md_tol',1e-6),('md_quad_2',15),('md_items',items[:-1]),('md_mode','Keşfedici (pozitif, üçgen)')]:
  fresh=page();fresh.session_state[view.KEY]=dict(saved);fresh.run();assert fresh.get('download_button')
  widget=next(w for kind in ('selectbox','number_input','multiselect') for w in fresh.get(kind) if w.key==control)
  widget.set_value(value).run();assert not fresh.exception and not fresh.get('download_button') and view.KEY not in fresh.session_state
 changed=data.copy();changed.iloc[0,1]=1-changed.iloc[0,1]
 at.session_state['raw_df']=changed;at.run();assert not at.exception and not at.get('download_button')
 assert len(calls)==1


def test_nonconvergence_suppresses_scores_curves_and_png(tmp_path):
 cfg,data,_,_=inputs('GRM_2D',tmp_path/'data')
 result=report(fit(data.head(100)+1,'GRM',cfg['pattern'],n_quadpts=5,max_iter=1))
 assert not result['metadata']['Converged']
 assert result['metadata']['Labels']==('1','2','3','4')
 assert not {'EAP','MAP','Curves','IIF','TIF','SEM'}&result['tables'].keys()
 bundle=downloads(result)
 assert bundle['png'] is None
 with zipfile.ZipFile(io.BytesIO(bundle['zip'])) as z:
  assert 'surfaces.png' not in z.namelist() and 'EAP.csv' not in z.namelist()
