"""Export contracts at real fixed native parameters: no fit, no inferential certification."""
import copy,io,json,zipfile
import numpy as np
import pandas as pd
import pytest
from reports.irt_multidim_report import downloads,figure,export_fingerprint
from validation.mirt_stage1.compare import inputs
from core.irt_multidim import surface_tables


@pytest.fixture
def result(tmp_path):
 cfg,data,engine,_=inputs('2PL_2D',tmp_path/'fixed')
 tables=surface_tables(engine,list(data.columns),('0','1'),2)
 p=engine.parameters
 tables['Parameters']=pd.DataFrame(dict(Item=data.columns,a1=p['slopes'][:,0],a2=p['slopes'][:,1],d=p['intercepts']))
 tables['Diagnostic']=pd.DataFrame({'Coordinate':['F1','F2'],'Value':[.123456789,-.234567891]})
 tables['Summary']=pd.DataFrame({'Setting':['Status','Additional_diagnostic'],'Value':['STALE LABEL','preserve me']})
 meta=dict(Model='2PL',Status='EXPERIMENTAL / NOT CERTIFIED',Inferential_status='WITHHELD / NOT CERTIFIED',
           Numerical_status='Fixed native probability/information fixture only',Engine='Python mirt 1.2.0',
           Converged=True,Dimensions=2,Limitations='Türkçe uyarı: posterior SD is not frequentist SE; no p/CI.',
           Applicable_conditions={'dimensions':2,'structure':'declared CFA only'},
           Unverifiable_assumptions='Latent normality is not verified from user data.',
           Fixture='FIXED PARAMETER UNIT TEST; NO CALIBRATION')
 return {'metadata':meta,'tables':tables}


def assert_original_columns(source,exported):
 for col in source:
  assert col in exported
  if pd.api.types.is_numeric_dtype(source[col]):
   np.testing.assert_allclose(exported[col].to_numpy(float),source[col].to_numpy(float),rtol=1e-12,atol=1e-13,equal_nan=True)
  else:assert exported[col].astype(str).tolist()==source[col].astype(str).tolist()


def test_csv_xlsx_zip_keep_numbers_status_scope_and_new_diagnostics(result):
 snapshot=copy.deepcopy(result);bundle=downloads(result)
 csv=pd.read_csv(io.BytesIO(bundle['csv']))
 assert_original_columns(result['tables']['Parameters'],csv)
 for column,key in [('Report_Status','Status'),('Report_Inferential_Status','Inferential_status'),
                    ('Report_Limitations','Limitations'),('Report_Engine','Engine')]:
  assert csv[column].eq(result['metadata'][key]).all()
 assert csv['Report_Applicable_Conditions'].str.contains('declared CFA').all()
 assert csv['Report_Unverifiable_Assumptions'].eq(result['metadata']['Unverifiable_assumptions']).all()
 sheets=pd.read_excel(io.BytesIO(bundle['xlsx']),sheet_name=None)
 assert set(result['tables'])|{'ScientificScope'}<=set(sheets)
 assert_original_columns(result['tables']['Parameters'],sheets['Parameters'])
 assert_original_columns(result['tables']['Diagnostic'],sheets['Diagnostic'])
 summary=dict(zip(sheets['Summary'].Setting,sheets['Summary'].Value))
 scope=dict(zip(sheets['ScientificScope'].Setting,sheets['ScientificScope'].Value))
 assert summary['Status']==scope['Status']==result['metadata']['Status']
 assert summary['Additional_diagnostic']=='preserve me'
 assert scope['Inferential_status']==result['metadata']['Inferential_status']
 assert scope['Limitations']==result['metadata']['Limitations']
 with zipfile.ZipFile(io.BytesIO(bundle['zip'])) as z:
  assert set(result['tables'])|{'ScientificScope'}=={n[:-4] for n in z.namelist() if n.endswith('.csv')}
  assert z.read('results.xlsx')==bundle['xlsx']
  assert json.loads(z.read('metadata.json'))==result['metadata']
  for name,source in result['tables'].items():
   exported=pd.read_csv(z.open(name+'.csv'))
   if name!='Summary':assert_original_columns(source,exported)
   assert exported['Report_Status'].eq(result['metadata']['Status']).all()
   assert exported['Report_Inferential_Status'].eq(result['metadata']['Inferential_status']).all()
   assert exported['Report_Limitations'].eq(result['metadata']['Limitations']).all()
 assert result['metadata']['Limitations'] in bundle['report'].decode()
 assert result['metadata']==snapshot['metadata']
 for name in result['tables']:pd.testing.assert_frame_equal(result['tables'][name],snapshot['tables'][name])


def test_missing_summary_and_existing_scope_are_preserved(result):
 result['tables'].pop('Summary')
 supplied=pd.DataFrame({'Output':['Posterior SD'],'Status':['CONDITIONAL POSTERIOR; NO COVERAGE CLAIM']})
 result['tables']['ScientificScope']=supplied
 sheets=pd.read_excel(io.BytesIO(downloads(result)['xlsx']),sheet_name=None)
 pd.testing.assert_frame_equal(sheets['ScientificScope'],supplied)
 assert {'Summary','ReportMetadata'}<=set(sheets)
 assert dict(zip(sheets['Summary'].Setting,sheets['Summary'].Value))['Status']==result['metadata']['Status']


def test_export_fingerprint_tracks_exact_values_metadata_and_selection(result):
 key=export_fingerprint(result,'M1','1')
 assert key==export_fingerprint(result,'M1','1')
 assert key!=export_fingerprint(result,'M2','1')
 assert key!=export_fingerprint(result,'M1','0')
 changed=copy.deepcopy(result)
 changed['tables']['Parameters'].loc[0,'d']=np.nextafter(changed['tables']['Parameters'].loc[0,'d'],np.inf)
 assert key!=export_fingerprint(changed,'M1','1')
 changed=copy.deepcopy(result);changed['tables']['Parameters'].columns.name='Changed axis name'
 # Column labels/dtypes/order are part of the key; index identity is hashed with the data.
 changed['tables']['Parameters']=changed['tables']['Parameters'].rename(columns={'d':'intercept'})
 assert key!=export_fingerprint(changed,'M1','1')
 changed=copy.deepcopy(result);changed['metadata']['Status']='WITHHELD / EVIDENCE WITHDRAWN'
 assert key!=export_fingerprint(changed,'M1','1')
 csv=pd.read_csv(io.BytesIO(downloads(changed)['csv']))
 assert csv['Report_Status'].eq('WITHHELD / EVIDENCE WITHDRAWN').all()
 assert_original_columns(result['tables']['Parameters'],csv)


def test_nonconvergence_and_unknown_convergence_withhold_png(result):
 for flag in (False,None,'False'):
  result['metadata']['Converged']=flag
  bundle=downloads(result,'M1','1')
  assert bundle['png'] is None
  with zipfile.ZipFile(io.BytesIO(bundle['zip'])) as z:assert 'surfaces.png' not in z.namelist()


def test_png_uses_actual_status_and_warnings(result):
 import matplotlib.pyplot as plt
 from PIL import Image
 item=result['tables']['Parameters'].Item.iloc[0]
 fig=figure(result,item,'1')
 assert fig._suptitle.get_text()==result['metadata']['Status']
 assert result['metadata']['Inferential_status'] in fig._supxlabel.get_text()
 assert 'Türkçe uyarı' in fig._supxlabel.get_text()
 plt.close(fig)
 bundle=downloads(result,item,'1')
 image=Image.open(io.BytesIO(bundle['png']));image.verify()
 with zipfile.ZipFile(io.BytesIO(bundle['zip'])) as z:assert z.read('surfaces.png')==bundle['png']


def test_csv_annotation_never_overwrites_existing_numeric_columns(result):
 result['tables']['Parameters']['Report_Status']=np.arange(len(result['tables']['Parameters']),dtype=float)
 bundle=downloads(result);csv=pd.read_csv(io.BytesIO(bundle['csv']))
 np.testing.assert_array_equal(csv['Report_Status'],result['tables']['Parameters']['Report_Status'])
 assert csv['_Report_Status'].eq(result['metadata']['Status']).all()
