"""Public Python mirt multidimensional adapters; no local calibration algorithm.

Constrained families and canonical gauges remain experimental; independent
profile evidence does not certify arbitrary data or inferential SE/CI/p.
"""
import itertools
import numpy as np
import pandas as pd
from core.irt_b14 import prepare_polytomous
from core.irt_python_mirt_adapter import normalize_solver_settings

STATUS='EXPERIMENTAL / NOT CERTIFIED'
MODELS=('Rasch','1PL (a=1)','2PL','GRM','PCM','GPCM','NRM','Bifactor')
LIMITATION=('Bağımsız R kabulü yalnız kayıtlı profiller içindir; yeni veride sonuç EXPERIMENTAL / NOT CERTIFIED. '
 'Ortalama 0, varyans 1; doğrulayıcı yapıda boyut başına ≥3 saf madde. '
 'Rasch/1PL(a=1), 2PL üzerinde; PCM, GPCM üzerinde sabit birim yüklerle kurulur. '
 'Ortak serbest-a 1PL, çok boyutlu 3PL/4PL/RSM desteklenmez. '
 'Parametre SE/GA/p, S-X²/M2 ve biçimsel model karşılaştırması kapalıdır. '
 'EAP SD posterior belirsizliğidir; Fisher SEM farklıdır. '
 'Nominal kategori kodları başarı sırası değildir; Wright Map/DTF yoktur. '
 'Keşfedici serbest rotasyon yerine tanımlanmış yapı kullanılır; 4–5D profiller deneysel ve bellek sınırlıdır.')


def support_matrix():
 return pd.DataFrame([dict(Model=m,Multidimensional_API=api,Phase1=m in ('2PL','GRM'),
   Status=STATUS if api else 'DISABLED',Note=note)
  for m,api,note in [('2PL',True,'Public slope/intercept CFA'),('GRM',True,'d_k=-sum(a)*b_k'),
  ('GPCM',True,'Public multidimensional GPCM'),('NRM',True,'Public NRM + fixed loading masks; R full nominal via gpcm_mats'),
  ('Rasch',True,'2PL with all slopes fixed to loading pattern'),('1PL (a=1)',True,'Same fixed-unit model as Rasch; not shared-free-a'),
  ('PCM',True,'GPCM with discrimination fixed to loading pattern'),('Bifactor',True,'Public bfactor; 3–4 specific groups, ≥3 items each'),
  ('1PL common free a',False,'Cross-coordinate shared-slope equality not in public constraints'),
  ('3PL',False,'Family rejects >1 factor'),('4PL',False,'Family rejects >1 factor'),
  ('RSM',False,'Multidimensional shared-step+location constraint not exposed')]])


def structure(pattern,correlated=False,exploratory=False):
 pattern=np.asarray(pattern)
 if pattern.ndim!=2 or pattern.shape[1] not in (2,3,4,5) or not np.isin(pattern,[0,1]).all():
  raise ValueError('2–5 sütunlu 0/1 madde-boyut matrisi gerekli')
 pattern=pattern.astype(bool);q=pattern.shape[1]
 if not pattern.any(1).all() or np.linalg.matrix_rank(pattern.astype(float))!=q:
  raise ValueError('Her madde yüklenmeli; yapı tam sütun ranklı olmalı')
 if exploratory:
  expected=np.ones_like(pattern);expected[:q,:]=np.tril(np.ones((q,q),bool))
  if correlated or len(pattern)<2*q+1 or not np.array_equal(pattern,expected):raise ValueError('Keşfedici profil: ortogonal, pozitif ve alt-üçgen tanımlama gerekli')
 elif any(np.count_nonzero(pattern[:,j] & (pattern.sum(1)==1))<3 for j in range(q)):
  raise ValueError('Rotasyon/sıra tanımlaması için her boyutta en az üç saf madde gerekli')
 lines=[f'F{j+1} = '+','.join(str(i+1) for i in np.flatnonzero(pattern[:,j])) for j in range(q)]
 if correlated=='full':lines.append('COV = '+'*'.join(f'F{j+1}' for j in range(q)))
 elif correlated:lines.append('COV = F1*F2')
 return '\n'.join(lines)


def nominal_start(pattern,categories):
 pattern=np.asarray(pattern);q=pattern.shape[1]
 values=np.zeros((len(pattern),categories,q));values[:,1,:]=pattern
 for c in range(2,categories):values[:,c,:]=pattern*(.8*np.sin(c*(np.arange(q)+1)))
 return values


def _canonical(f,covariance,pattern,model):
 from copy import copy
 q=pattern.shape[1];signs=np.ones(q)
 if model=='NRM':
  slopes=f.model.parameters['slopes']
  for j in range(q):
   first=np.flatnonzero(pattern[:,j] & (pattern.sum(1)==1))[0]
   value=slopes[first,1,j]
   if abs(value)<np.sqrt(np.finfo(float).eps):raise ValueError('Nominal referans kontrastı sayısal olarak sıfır; yön tanımlanamadı')
   signs[j]=np.sign(value)
 elif model=='Bifactor':
  a=f.model.get_loading_matrix()
  for j in range(q):
   value=a[np.flatnonzero(pattern[:,j])[0],j]
   if abs(value)<np.sqrt(np.finfo(float).eps):raise ValueError('Bifactor referans yükü sıfır; yön tanımlanamadı')
   signs[j]=np.sign(value)
 if np.any(signs!=1):
  original=f;f=copy(f);f.model=original.model.copy()
  if model=='NRM':f.model.set_parameters(slopes=original.model.parameters['slopes']*signs)
  else:
   group=np.argmax(pattern[:,1:],axis=1)
   f.model.set_parameters(general_loadings=original.model.parameters['general_loadings']*signs[0],
     specific_loadings=original.model.parameters['specific_loadings']*signs[group+1])
  covariance=covariance*np.outer(signs,signs);f.latent_covariance=covariance;f.latent_mean=np.zeros(q)
  theta=np.vstack((np.zeros(q),np.eye(q),-np.eye(q)))
  np.testing.assert_allclose(f.model.probability(theta),original.model.probability(theta*signs),atol=1e-12)
 return f,covariance,signs


def fit(data,model,pattern,*,correlated=False,n_quadpts=15,max_iter=1000,tol=1e-7,labels=None,exploratory=False):
 import mirt
 if model not in MODELS:raise ValueError('Desteklenmeyen MIRT ailesi')
 data=pd.DataFrame(data);groups=None
 if exploratory and model not in ('2PL','GRM','GPCM'):raise ValueError('Bu aile için keşfedici profil doğrulanmadı')
 if model=='Bifactor':
  groups=np.asarray(pattern)
  if groups.ndim!=1 or not np.equal(groups,np.floor(groups)).all() or (groups<0).any():raise ValueError('Madde başına geçerli özgül grup kodu gerekli')
  unique=np.unique(groups)
  if len(unique) not in (3,4) or any(np.count_nonzero(groups==g)<3 for g in unique):raise ValueError('Bifactor: 3–4 özgül grup ve grup başına ≥3 madde gerekli')
  if correlated:raise ValueError('Bifactor faktörleri ortogonal olmalı')
  pattern=np.column_stack((np.ones(len(groups),bool),groups[:,None]==unique));q=len(unique)+1
  syntax='G + '+','.join(f'S{g}' for g in unique)
 else:
  syntax=structure(pattern,correlated,exploratory);pattern=np.asarray(pattern,bool);q=pattern.shape[1]
 if len(data.columns)!=len(pattern) or not data.columns.is_unique:raise ValueError('Madde-boyut matrisi veriyle uyuşmuyor')
 if len(data)<30 or data.isna().any().any():raise ValueError('En az 30 eksiksiz satır gerekli; otomatik atama/silme yok')
 if model in ('GRM','GPCM','PCM','NRM'):
  x,labels=prepare_polytomous(data,'GPCM' if model=='PCM' else model,categories=labels if model=='NRM' else None)
 else:
  x=data.apply(pd.to_numeric,errors='raise');labels=('0','1')
  if not np.isfinite(x.to_numpy(float)).all() or not x.isin([0,1]).all().all():raise ValueError('İkili model için eksiksiz 0/1 veri')
  x=x.astype(int)
 if any(set(x[c])!=set(range(len(labels))) for c in x):raise ValueError('Her maddede bütün kategoriler gözlenmeli')
 settings=normalize_solver_settings(max_iter=max_iter,tol=tol,n_quadpts=n_quadpts)
 if n_quadpts<5 or n_quadpts**q>4000 or len(x)*x.shape[1]*n_quadpts**q>20_000_000:
  raise ValueError('GH düğüm/bellek sınırı: ≥5/eksen, toplam ≤4000, N×J×düğüm ≤20 milyon')
 covariance=np.eye(q)
 if model=='Bifactor':
  result=mirt.bfactor(x,groups,n_quadpts=n_quadpts,max_iter=max_iter,tol=tol,compute_standard_errors=False,
   start_values=dict(general_loadings=np.full(len(x.columns),.7),specific_loadings=np.full(len(x.columns),.5),intercepts=np.zeros(len(x.columns))))
 elif model=='NRM':
  from mirt.models.polytomous import NominalResponseModel
  from mirt.estimation.latent_density import FactorCovarianceDensity
  engine=NominalResponseModel(x.shape[1],len(labels),n_factors=q,item_names=list(x.columns))
  mask=np.broadcast_to(pattern[:,None,:],(x.shape[1],len(labels),q)).copy();mask[:,0,:]=False
  intercept_mask=np.ones((x.shape[1],len(labels)),bool);intercept_mask[:,0]=False
  engine.set_free_parameter_masks(dict(slopes=mask,intercepts=intercept_mask))
  density=None
  if correlated:
   free=~np.eye(q,dtype=bool) if correlated=='full' else np.zeros((q,q),bool)
   if correlated!='full':free[0,1]=free[1,0]=True
   density=FactorCovarianceDensity(q,free=free)
  estimator=mirt.EMEstimator(**settings,latent_density=density,compute_standard_errors=False,
   item_optim_ftol=1e-10,item_optim_maxiter=200)
  result=estimator.fit(engine,x.to_numpy(),start=dict(slopes=nominal_start(pattern,len(labels)),intercepts=np.zeros((x.shape[1],len(labels)))))
 else:
  unit=model in ('Rasch','1PL (a=1)','PCM');binary=model in ('2PL','Rasch','1PL (a=1)')
  family='2PL' if binary else 'GRM' if model=='GRM' else 'GPCM'
  slopes=pattern.astype(float)*(1 if unit else 1.2)
  key='slopes' if binary else 'discrimination'
  start={key:slopes,'intercepts':np.zeros(len(pattern))} if binary else {
   key:slopes,'thresholds' if model=='GRM' else 'steps':np.tile(np.linspace(-1,1,len(labels)-1),(len(pattern),1))}
  result=mirt.fit_mirt(x,model=family,spec=mirt.mirt_model(syntax,item_names=list(x.columns)),
   n_factors=q,n_categories=len(labels) if not binary else None,start_values=start,
   fixed={key:True} if unit else None,priors=None,compute_standard_errors=False,accelerate='none',**settings)
  actual=np.asarray(result.model.parameters[key])
  if np.any(actual[~pattern]!=0) or np.any(actual[pattern]<=0):raise ValueError('Yük yapısı/işaret tanımlaması bozuldu')
  if unit and not np.array_equal(actual,pattern.astype(float)):raise ValueError('Birim yük kısıtı bozuldu')
 raw=result
 covariance=np.eye(q) if result.latent_covariance is None else np.asarray(result.latent_covariance)
 if not np.allclose(np.diag(covariance),1,atol=1e-12) or np.linalg.eigvalsh(covariance).min()<=0:
  raise ValueError('Birim varyans/pozitif tanımlı korelasyon kısıtı bozuldu')
 result,covariance,signs=_canonical(result,covariance,pattern,model)
 return dict(fit=result,raw_fit=raw,coded=x,labels=labels,pattern=pattern,covariance=covariance,
  model=model,syntax=syntax,status=STATUS,settings=settings,correlated=correlated,orientation=signs.tolist(),exploratory=exploratory)


def report(outcome,person_ids=None):
 import mirt
 f=outcome['fit'];x=outcome['coded'];q=outcome['pattern'].shape[1];k=x.shape[1]
 covariance=outcome['covariance'];params=f.model.parameters
 if outcome['model']=='NRM':
  rows=[];a=params['slopes']
  for i,item in enumerate(x.columns):
   for c,label in enumerate(outcome['labels']):rows.append(dict(Item=item,Category=label,Reference=c==0,
    d=params['intercepts'][i,c],**{f'a{j+1}':a[i,c,j] for j in range(q)}))
  table=pd.DataFrame(rows)
 else:
  a=f.model.get_loading_matrix() if outcome['model']=='Bifactor' else np.asarray(params['slopes' if outcome['model'] in ('2PL','Rasch','1PL (a=1)') else 'discrimination'])
  table=pd.DataFrame(a,index=x.columns,columns=[f'a{j+1}' for j in range(q)])
  if outcome['model'] in ('2PL','Rasch','1PL (a=1)','Bifactor'):table['d']=params['intercepts']
  else:
   key='thresholds' if outcome['model']=='GRM' else 'steps'
   for j in range(len(outcome['labels'])-1):
    table[f'b{j+1}']=params[key][:,j];table[f'd{j+1}']=-a.sum(1)*params[key][:,j]
 frames={'Parameters':table if outcome['model']=='NRM' else table.reset_index(names='Item'),
  'Pattern':pd.DataFrame(outcome['pattern'].astype(int),index=x.columns,columns=[f'F{j+1}' for j in range(q)]).reset_index(names='Item'),
  'Covariance':pd.DataFrame(covariance,columns=[f'F{j+1}' for j in range(q)]).reset_index(names='Factor')}
 metadata=dict(Model=outcome['model'],Status=STATUS,Engine='Python mirt '+mirt.__version__,
  N=len(x),Items=k,Dimensions=q,Converged=bool(f.converged),Iterations=f.n_iterations,LogLik=f.log_likelihood,
  AIC=f.aic,BIC=f.bic,Parameters=f.n_parameters,Syntax=outcome['syntax'],Labels=outcome['labels'],
  Orientation=outcome.get('orientation'),Exploratory=outcome.get('exploratory',False),SE='disabled: multidimensional auto complete_data is not observed-information SE',
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
