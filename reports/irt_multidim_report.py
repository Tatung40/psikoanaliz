"""Safe multidimensional exports; no inferential statistics are manufactured."""
import hashlib,io,json,textwrap,zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reports.irt_b14_report import safe_frame,export_xlsx


def figure(result,item,category):
 tables=result['tables'];fig,grid=plt.subplots(2,2,figsize=(11,8),layout='constrained');axes=grid.flat
 p=tables['Curves'];p=p[(p.Item==item)&(p.Category.astype(str)==str(category))]
 z=p.pivot(index='Theta2',columns='Theta1',values='Probability')
 im=axes[0].pcolormesh(z.columns,z.index,z.to_numpy(),vmin=0,vmax=1,shading='auto');fig.colorbar(im,ax=axes[0])
 axes[0].set(title=f'{item}: category {category}',xlabel='F1',ylabel='F2')
 n=tables['SEM'];cut=n[np.isclose(n.Theta2,0)]
 curves=tables['Curves'];curves=curves[(curves.Item==item)&np.isclose(curves.Theta2,0)]
 for label,part in curves.groupby('Category'):
  part=part.sort_values('Theta1');axes[1].plot(part.Theta1,part.Probability,label=str(label))
 axes[1].set(title=f'{item}: category probability cuts',xlabel='F1',ylim=(0,1));axes[1].legend()
 for ax,t,title in [(axes[2],tables['IIF'].query('Item == @item'),f'{item}: IIF diagonal'),(axes[3],tables['TIF'],'TIF diagonal')]:
  for j in sorted(t.Row.unique()):
   part=t[(t.Row==j)&(t.Column==j)].merge(cut[['Node','Theta1']],on='Node').sort_values('Theta1')
   ax.plot(part.Theta1,part.Information,label=f'I{j}{j}')
  ax.set(title=title+'; F2…Fq = 0',xlabel='F1');ax.legend()
 m=result['metadata']
 fig.suptitle(textwrap.fill(str(m.get('Status','UNKNOWN / NOT CERTIFIED')),105),fontsize=11)
 footer=(str(m.get('Inferential_status',m.get('InferentialStatus','WITHHELD / NOT CERTIFIED')))+'; '+
         str(m.get('Limitations','Applicability and inferential validity not established.'))+
         ' Other dimensions are fixed at zero; diagonal Ijj curves are explicitly directional, not universal scalar information.')
 fig.supxlabel(textwrap.fill(footer,145),fontsize=6)
 return fig


_EXPORT_SCHEMA = 'mirt-shared-export-v1'


def _json_default(value):
 if isinstance(value,np.ndarray):return value.tolist()
 if isinstance(value,np.generic):return value.item()
 return str(value)


def _text(value):
 return value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,sort_keys=True,default=_json_default)


def export_fingerprint(result,item=None,category=None):
 """Exact content/metadata/selection key; no fitting, scoring or export generation."""
 digest=hashlib.sha256()
 def add(value):
  raw=json.dumps(value,ensure_ascii=False,sort_keys=True,default=_json_default,separators=(',',':')).encode()
  digest.update(str(len(raw)).encode()+b':'+raw)
 add([_EXPORT_SCHEMA,item,category,result['metadata']])
 for name,frame in result['tables'].items():
  add([name,list(frame.columns),[str(t) for t in frame.dtypes],frame.index.names])
  try:digest.update(pd.util.hash_pandas_object(frame,index=True,categorize=False).to_numpy().tobytes())
  except TypeError:add(frame.to_dict(orient='split'))
 return digest.hexdigest()


def _export_tables(result):
 """Authoritative metadata survives even if the caller omitted/staled its Summary."""
 m=result['metadata'];tables={name:frame.copy(deep=True) for name,frame in result['tables'].items()}
 rows=[dict(Setting=str(key),Value=_text(value)) for key,value in m.items()]
 rows.extend([dict(Setting='Export_schema',Value=_EXPORT_SCHEMA),
              dict(Setting='Inferential_status',Value=str(m.get('Inferential_status',m.get('InferentialStatus','WITHHELD / NOT CERTIFIED'))))])
 scope=pd.DataFrame(rows).drop_duplicates('Setting',keep='last')
 summary=tables.get('Summary')
 if summary is None:tables['Summary']=scope.copy()
 elif {'Setting','Value'}<=set(summary.columns):
  summary['Value']=summary['Value'].astype(object)
  for row in scope.itertuples(index=False):
   match=summary['Setting'].astype(str)==row.Setting
   if match.any():summary.loc[match,'Value']=row.Value
   else:summary=pd.concat([summary,pd.DataFrame([dict(Setting=row.Setting,Value=row.Value)])],ignore_index=True)
  tables['Summary']=summary
 if 'ScientificScope' not in tables:tables['ScientificScope']=scope
 else:
  # Preserve a caller-supplied output-level scope matrix; add authoritative report metadata separately.
  name='ReportMetadata'
  while name in tables:name='_'+name
  tables[name]=scope
 return tables


def _annotated(frame,metadata,full_scope=False):
 """A detached CSV must retain its actual status and methodological warning."""
 out=frame.copy()
 values={
  'Report_Status':metadata.get('Status','UNKNOWN / NOT CERTIFIED'),
  'Report_Inferential_Status':metadata.get('Inferential_status',metadata.get('InferentialStatus','WITHHELD / NOT CERTIFIED')),
  'Report_Limitations':metadata.get('Limitations','Applicability and inferential validity not established.'),
  'Report_Engine':metadata.get('Engine','UNRECORDED'),
 }
 if full_scope:
  values.update(Report_Applicable_Conditions=metadata.get('Applicable_conditions','Not established beyond the documented reference fixture.'),
                Report_Unverifiable_Assumptions=metadata.get('Unverifiable_assumptions','Latent data-generating assumptions are not directly verified from user data.'))
 for name,value in values.items():
  # Never overwrite a numeric or diagnostic column supplied by an engine.
  while name in out:name='_'+name
  out[name]=_text(value)
 return out


def downloads(result,item=None,category=None):
 tables=_export_tables(result);m=result['metadata']
 method=('Independent SciPy direct fixed-grid marginal ML; separate experimental estimator. ' if m.get('Backend')=='SCIPY_GENERAL' else 'Marginal ML, normal latent distribution, declared fixed zero loadings; no local estimator. ')
 note=('# Multidimensional IRT / '+str(m.get('Status','UNKNOWN / NOT CERTIFIED'))+'\n\n'+str(m.get('Limitations','Applicability and inferential validity not established.'))+
  '\n\n'+method+
  'Convergence is a numerical stopping result, not evidence of global optimality or model fit. '
  'EAP posterior SD is not a parameter SE. Joint Fisher SEM uses the inverse full test '
  'information matrix and is suppressed at numerical singularities. No SE/CI/p, S-X², '
  'Wright Map or inferential model comparisons. Nominal codes have no ordinal interpretation. '
  'AIC/BIC/logLik are descriptive; comparisons require the same people, categories, '
  'likelihood and quadrature. No automatic model selection.\n\n```json\n'+
  json.dumps(m,ensure_ascii=False,indent=2,default=_json_default)+'\n```\n')
 xlsx=export_xlsx(tables);buf=io.BytesIO();picture=None
 converged=m.get('Converged',False)
 if isinstance(converged,(bool,np.bool_)) and converged and item is not None and 'Curves' in tables:
  fig=figure(result,item,category)
  try:fig.savefig(buf,format='png',dpi=120);picture=buf.getvalue()
  finally:plt.close(fig)
 buf=io.BytesIO()
 with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
  for name,frame in tables.items():
   z.writestr(name+'.csv',safe_frame(_annotated(frame,m,full_scope=name=='Parameters')).to_csv(index=False))
  z.writestr('report.md',note);z.writestr('metadata.json',json.dumps(m,ensure_ascii=False,default=_json_default,indent=2));z.writestr('results.xlsx',xlsx)
  if picture:z.writestr('surfaces.png',picture)
 return dict(csv=safe_frame(_annotated(tables['Parameters'],m,full_scope=True)).to_csv(index=False).encode('utf-8-sig'),xlsx=xlsx,zip=buf.getvalue(),png=picture,report=note.encode('utf-8'))
