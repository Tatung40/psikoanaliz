"""Phase 1 public Python mirt CFA adapter. No local calibration algorithm.

Only identified 2D/3D 2PL/GRM profiles. Expanded families require Phase 1
independent R acceptance; SE/CI/p, exploratory rotations and bifactor disabled.
"""
import itertools
import numpy as np
import pandas as pd
from core.irt_b14 import prepare_polytomous
from core.irt_python_mirt_adapter import normalize_solver_settings

STATUS='EXPERIMENTAL / NOT CERTIFIED'
LIMITATION=('Aşama 1 bağımsız R kabulü tamamlanmadı. Yalnız 2D/3D doğrulayıcı 2PL/GRM; '
 'yükler pozitif, her boyutta en az üç saf madde, referans ortalama 0/varyans 1. '
 'İlişkili profilde yalnız F1–F2 korelasyonu serbesttir; diğer korelasyonlar 0. '
 'Gözlenen-parametre SE/GA/p ve uyum testleri kapalıdır. Tam-veri eğriliği SE olarak kullanılmaz. '
 'EAP SD posterior belirsizliğidir; Fisher SEM başka bir niceliktir. '
 'Keşfedici/bifactor ve diğer aileler Aşama 1 kabulüne kadar kapalıdır.')


def support_matrix():
 return pd.DataFrame([dict(Model=m,Multidimensional_API=api,Phase1=m in ('2PL','GRM'),
   Status=STATUS if m in ('2PL','GRM') else 'DISABLED / PHASE 1 PENDING',Note=note)
  for m,api,note in [('2PL',True,'spec: slope/intercept CFA'),('GRM',True,'d_k = -sum(a)*b_k'),
  ('GPCM',True,'API available; not validated/activated'),('NRM',True,'Raw multidimensional API; spec does not support NRM'),
  ('Rasch/1PL',False,'Public family disallows >1 factor'),('3PL',False,'Public family disallows >1 factor'),
  ('4PL',False,'Public family disallows >1 factor'),('PCM',False,'Public family disallows >1 factor'),
  ('RSM',False,'Not in public fit_mirt family factory')]])


def structure(pattern,correlated=False):
 pattern=np.asarray(pattern)
 if pattern.ndim!=2 or pattern.shape[1] not in (2,3) or not np.isin(pattern,[0,1]).all():
  raise ValueError('2/3 sütunlu 0/1 madde-boyut matrisi gerekli')
 pattern=pattern.astype(bool);q=pattern.shape[1]
 if not pattern.any(1).all() or np.linalg.matrix_rank(pattern.astype(float))!=q:
  raise ValueError('Her madde yüklenmeli; yapı tam sütun ranklı olmalı')
 if any(np.count_nonzero(pattern[:,j] & (pattern.sum(1)==1))<3 for j in range(q)):
  raise ValueError('Rotasyon/sıra tanımlaması için her boyutta en az üç saf madde gerekli')
 lines=[f'F{j+1} = '+','.join(str(i+1) for i in np.flatnonzero(pattern[:,j])) for j in range(q)]
 if correlated:lines.append('COV = F1*F2')
 return '\n'.join(lines)


def fit(data,model,pattern,*,correlated=False,n_quadpts=15,max_iter=1000,tol=1e-7):
 import mirt
 if model not in ('2PL','GRM'):raise ValueError('Aşama 1 yalnız 2PL/GRM')
 data=pd.DataFrame(data)
 syntax=structure(pattern,correlated);pattern=np.asarray(pattern,bool);q=pattern.shape[1]
 if len(data.columns)!=len(pattern) or not data.columns.is_unique:raise ValueError('Madde-boyut matrisi veriyle uyuşmuyor')
 if len(data)<30 or data.isna().any().any():raise ValueError('En az 30 eksiksiz satır gerekli; otomatik atama/silme yok')
 if model=='GRM':x,labels=prepare_polytomous(data,model)
 else:
  x=data.apply(pd.to_numeric,errors='raise');labels=('0','1')
  if not np.isfinite(x.to_numpy(float)).all() or not x.isin([0,1]).all().all():raise ValueError('2PL için eksiksiz 0/1 veri')
  x=x.astype(int)
 if any(set(x[c])!=set(range(len(labels))) for c in x):raise ValueError('Her maddede bütün kategoriler gözlenmeli')
 settings=normalize_solver_settings(max_iter=max_iter,tol=tol,n_quadpts=n_quadpts)
 if n_quadpts<5 or n_quadpts**q>4000 or len(x)*x.shape[1]*n_quadpts**q>20_000_000:
  raise ValueError('GH düğüm/bellek sınırı: ≥5/eksen, toplam ≤4000, N×J×düğüm ≤20 milyon')
 slopes=pattern.astype(float)*1.2
 start={'slopes':slopes,'intercepts':np.zeros(len(pattern))} if model=='2PL' else {
  'discrimination':slopes,'thresholds':np.tile(np.linspace(-1,1,len(labels)-1),(len(pattern),1))}
 result=mirt.fit_mirt(x,model=model,spec=mirt.mirt_model(syntax,item_names=list(x.columns)),
  n_factors=q,n_categories=len(labels) if model=='GRM' else None,start_values=start,priors=None,
  compute_standard_errors=False,accelerate='none',**settings)
 actual=np.asarray(result.model.parameters['slopes' if model=='2PL' else 'discrimination'])
 if np.any(actual[~pattern]!=0) or np.any(actual[pattern]<=0):raise ValueError('Yük yapısı/işaret tanımlaması bozuldu')
 covariance=np.eye(q) if result.latent_covariance is None else np.asarray(result.latent_covariance)
 if not np.allclose(np.diag(covariance),1,atol=1e-12) or np.linalg.eigvalsh(covariance).min()<=0:
  raise ValueError('Birim varyans/pozitif tanımlı korelasyon kısıtı bozuldu')
 return dict(fit=result,coded=x,labels=labels,pattern=pattern,covariance=covariance,
  model=model,syntax=syntax,status=STATUS,settings=settings,correlated=correlated)


def report(outcome,person_ids=None):
 import mirt
 f=outcome['fit'];x=outcome['coded'];q=outcome['pattern'].shape[1];k=x.shape[1]
 covariance=outcome['covariance'];params=f.model.parameters
 a=np.asarray(params['slopes' if outcome['model']=='2PL' else 'discrimination'])
 table=pd.DataFrame(a,index=x.columns,columns=[f'a{j+1}' for j in range(q)])
 if outcome['model']=='2PL':table['d']=params['intercepts']
 else:
  for j in range(len(outcome['labels'])-1):
   table[f'b{j+1}']=params['thresholds'][:,j]
   table[f'd{j+1}']=-a.sum(1)*params['thresholds'][:,j]
 frames={'Parameters':table.reset_index(names='Item'),
  'Pattern':pd.DataFrame(outcome['pattern'].astype(int),index=x.columns,columns=[f'F{j+1}' for j in range(q)]).reset_index(names='Item'),
  'Covariance':pd.DataFrame(covariance,columns=[f'F{j+1}' for j in range(q)]).reset_index(names='Factor')}
 metadata=dict(Model=outcome['model'],Status=STATUS,Engine='Python mirt '+mirt.__version__,
  N=len(x),Items=k,Dimensions=q,Converged=bool(f.converged),Iterations=f.n_iterations,LogLik=f.log_likelihood,
  AIC=f.aic,BIC=f.bic,Parameters=f.n_parameters,Syntax=outcome['syntax'],Labels=outcome['labels'],
  SE='disabled: multidimensional auto complete_data is not observed-information SE',
  Limitations=LIMITATION,**outcome['settings'])
 frames['Summary']=pd.DataFrame([dict(Setting=n,Value=str(v)) for n,v in metadata.items()])
 if not f.converged:return dict(metadata=metadata,tables=frames)
 ids=list(x.index) if person_ids is None else list(person_ids)
 for method in ('EAP','MAP'):
  score=mirt.fscores(f,x.to_numpy(),method=method,n_quadpts=outcome['settings']['n_quadpts'],person_ids=ids)
  rows=pd.DataFrame(np.asarray(score.theta),columns=[f'F{j+1}' for j in range(q)]);rows.insert(0,'ID',ids)
  if method=='EAP':
   for j in range(q):rows[f'Posterior_SD_F{j+1}']=np.asarray(score.standard_error)[:,j]
  frames[method]=rows
 # Shared F1/F2 surface; all other dimensions held exactly at zero.
 grid=np.array(list(itertools.product(np.linspace(-3,3,17),repeat=2)))
 theta=np.zeros((len(grid),q));theta[:,:2]=grid
 p=np.asarray(f.model.probability(theta))
 if p.ndim==2:p=np.stack((1-p,p),-1)
 if p.shape!=(len(theta),k,len(outcome['labels'])) or not np.isfinite(p).all() or not np.allclose(p.sum(-1),1,atol=1e-10):
  raise ValueError('Geçersiz çok boyutlu kategori olasılıkları')
 frames['Curves']=pd.DataFrame([dict(Node=n,Theta1=grid[n,0],Theta2=grid[n,1],Item=item,Category=label,Probability=p[n,j,c])
  for n in range(len(theta)) for j,item in enumerate(x.columns) for c,label in enumerate(outcome['labels'])])
 information=np.stack([f.model.item_information_matrix(theta,j) for j in range(k)],axis=1)
 if not np.isfinite(information).all():raise ValueError('Geçersiz Fisher bilgi matrisi')
 frames['IIF']=pd.DataFrame([dict(Node=n,Item=item,Row=r+1,Column=c+1,Information=information[n,j,r,c])
  for n in range(len(theta)) for j,item in enumerate(x.columns) for r in range(q) for c in range(q)])
 total=f.model.test_information_matrix(theta)
 np.testing.assert_allclose(total,information.sum(1),atol=1e-10)
 rows=[]
 for n,matrix in enumerate(total):
  eig=np.linalg.eigvalsh(matrix);stable=eig.min()>np.finfo(float).eps*q*max(eig.max(),1)
  sem=np.sqrt(np.diag(np.linalg.solve(matrix,np.eye(q)))) if stable else np.full(q,np.nan)
  rows.append(dict(Node=n,Theta1=grid[n,0],Theta2=grid[n,1],Stable=bool(stable),**{f'Joint_Fisher_SEM_F{j+1}':sem[j] for j in range(q)}))
 frames['SEM']=pd.DataFrame(rows)
 frames['TIF']=pd.DataFrame([dict(Node=n,Row=r+1,Column=c+1,Information=total[n,r,c]) for n in range(len(theta)) for r in range(q) for c in range(q)])
 return dict(metadata=metadata,tables=frames)
