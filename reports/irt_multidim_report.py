"""Safe multidimensional exports; no inferential statistics are manufactured."""
import io,json,zipfile
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
 fig.suptitle('EXPERIMENTAL / NOT CERTIFIED — other dimensions fixed at zero')
 return fig


def downloads(result,item=None,category=None):
 tables=result['tables'];m=result['metadata']
 note=('# Multidimensional IRT / '+m['Status']+'\n\n'+m['Limitations']+
  '\n\nMarginal ML, normal latent distribution, declared fixed zero loadings; no local estimator. '
  'Convergence is a numerical stopping result, not evidence of global optimality or model fit. '
  'EAP posterior SD is not a parameter SE. Joint Fisher SEM uses the inverse full test '
  'information matrix and is suppressed at numerical singularities. No SE/CI/p, S-X², '
  'Wright Map or inferential model comparisons. Nominal codes have no ordinal interpretation. '
  'AIC/BIC/logLik are descriptive; comparisons require the same people, categories, '
  'likelihood and quadrature. No automatic model selection.\n\n```json\n'+
  json.dumps(m,ensure_ascii=False,indent=2,default=str)+'\n```\n')
 xlsx=export_xlsx(tables);buf=io.BytesIO();picture=None
 if m['Converged'] and item is not None:
  fig=figure(result,item,category);fig.savefig(buf,format='png',dpi=120);plt.close(fig);picture=buf.getvalue()
 buf=io.BytesIO()
 with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
  for name,frame in tables.items():z.writestr(name+'.csv',safe_frame(frame).to_csv(index=False))
  z.writestr('report.md',note);z.writestr('metadata.json',json.dumps(m,ensure_ascii=False,default=str,indent=2));z.writestr('results.xlsx',xlsx)
  if picture:z.writestr('surfaces.png',picture)
 return dict(csv=safe_frame(tables['Parameters']).to_csv(index=False).encode('utf-8-sig'),xlsx=xlsx,zip=buf.getvalue(),png=picture,report=note.encode())
