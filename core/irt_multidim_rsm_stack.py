"""Exact between-item orthogonal RSM using ONE public scalar shared-step fit.

Each original response occurs once; off-block cells are structural missingness.
"""
import numpy as np
import pandas as pd
import mirt
from mirt.models.polytomous import RatingScaleModel
from core.irt_b14 import prepare_polytomous
from core.irt_multidim import STATUS,SURFACE_POINTS,surface_tables
from core.irt_python_mirt_adapter import normalize_solver_settings

NAME='RSM — ortak adımlar, ortogonal bloklar'
MODELS=(NAME,)
LIMITATION=('EXPERIMENTAL / NOT CERTIFIED. Yalnız ortogonal between-item, birim yükler, bağımsız N(0,1) boyutlar. '
 'Tüm boyutlarda TEK ortak adım vektörü, TEK gerçek Python mirt RatingScaleModel fitinde birlikte kestirilir. '
 'Yapısal eksik bloklu satır temsili tam aynı ortak olabilirliği verir; bağımsız RSM fitleri veya GPCM etiketi değildir. '
 'İlk ortak adım 0 sabitlenir; diğer ortak adımlar ve madde konumları kestirilir. '
 'Korelasyon/within-item, parametre SE/GA/p ve doğrulanmamış uyum/LR istatistikleri kapalıdır. '
 'Kişi SD/SEM item parametreleri koşulundadır; ortak adımların kestirim belirsizliğini içermez. '
 'BIC özgün kişi sayısını kullanır; native yapısal satır BIC ayrıca korunur.')


def validate(data,pattern,correlated=False,exploratory=False):
 data=pd.DataFrame(data);p=np.asarray(pattern)
 if p.ndim!=2 or p.shape[1] not in (2,3,4,5) or len(p)!=len(data.columns) or not np.isin(p,[0,1]).all():raise ValueError('Geçerli 2–5D madde-boyut matrisi gerekli')
 if correlated or exploratory or not np.all(p.sum(1)==1):raise ValueError('RSM yalnız ortogonal between-item yapı destekler')
 if any(p[:,d].sum()<6 for d in range(p.shape[1])):raise ValueError('RSM boyut başına en az 6 madde gerekli; bu yeterlilik sertifikası değildir')
 if len(data)<30 or not data.columns.is_unique or data.isna().any().any():raise ValueError('En az 30 eksiksiz kişi ve benzersiz maddeler gerekli')
 x,labels=prepare_polytomous(data,'GPCM')
 if len(labels)<3 or any(set(x[c])!=set(range(len(labels))) for c in x):raise ValueError('RSM: en az 3 kategori; her maddede tüm kategoriler gözlenmeli')
 if SURFACE_POINTS**2*len(p)*(p.shape[1]**2+len(labels))>300_000:raise ValueError('RSM rapor bellek bütçesi aşıldı')
 return x,p.astype(bool),labels


def stack_responses(data,pattern):
 values=np.asarray(data,dtype=int);n,j=values.shape;q=np.asarray(pattern).shape[1];out=np.full((n*q,j),-1,dtype=int)
 for d in range(q):
  index=np.flatnonzero(np.asarray(pattern)[:,d]);out[d*n:(d+1)*n,index]=values[:,index]
 if np.count_nonzero(out>=0)!=values.size:raise ValueError('Yapısal satırlama yanıt sayısını değiştirdi')
 return out


class Curves:
 def __init__(self,model,pattern,categories):self.model=model;self.pattern=np.asarray(pattern,bool);self.categories=categories
 def probability(self,theta):
  theta=np.asarray(theta);out=np.empty((len(theta),len(self.pattern),self.categories))
  for d in range(self.pattern.shape[1]):
   index=np.flatnonzero(self.pattern[:,d]);out[:,index,:]=self.model.probability(theta[:,d,None])[:,index,:]
  return out
 def item_information_matrix(self,theta,item_idx):
  d=int(np.flatnonzero(self.pattern[item_idx])[0]);out=np.zeros((len(theta),self.pattern.shape[1],self.pattern.shape[1]))
  out[:,d,d]=np.asarray(self.model.information(np.asarray(theta)[:,d,None],item_idx=item_idx)).reshape(-1);return out
 def test_information_matrix(self,theta):
  out=np.zeros((len(theta),self.pattern.shape[1],self.pattern.shape[1]))
  for d in range(self.pattern.shape[1]):out[:,d,d]=sum(self.model.information(np.asarray(theta)[:,d,None],item_idx=int(j)) for j in np.flatnonzero(self.pattern[:,d]))
  return out


def fit(data,model,pattern,*,correlated=False,exploratory=False,n_quadpts=61,max_iter=1000,tol=1e-10):
 if model not in MODELS:raise ValueError('Doğrulanmamış RSM yolu')
 x,p,labels=validate(data,pattern,correlated,exploratory);settings=normalize_solver_settings(max_iter=max_iter,tol=tol,n_quadpts=n_quadpts)
 if n_quadpts<5 or n_quadpts>81 or len(x)*p.shape[1]*len(x.columns)*len(labels)*n_quadpts>20_000_000:raise ValueError('RSM yapısal satır/GH bellek bütçesi aşıldı')
 native=RatingScaleModel(len(x.columns),len(labels),item_names=list(x.columns));start=dict(difficulty=np.zeros(len(x.columns)),thresholds=np.arange(len(labels)-1,dtype=float))
 native.set_parameters(**start)
 estimator=mirt.EMEstimator(**settings,compute_standard_errors=False,item_optim_ftol=1e-13,item_optim_maxiter=200,accelerate='none')
 stacked=stack_responses(x,p);f=estimator.fit(native,stacked,start=start)
 if f.model.parameters['thresholds'][0]!=0 or f.n_parameters!=len(x.columns)+len(labels)-2:raise ValueError('RSM ortak adım tanımlaması veya parametre sayısı bozuldu')
 return dict(fit=f,coded=x,pattern=p,labels=labels,settings=settings,stacked_rows=len(stacked),model=model)


def report(outcome,person_ids=None):
 f=outcome['fit'];x=outcome['coded'];p=outcome['pattern'];labels=outcome['labels'];q=p.shape[1];n=len(x);j=len(x.columns);params=f.model.parameters
 ids=list(x.index) if person_ids is None else list(person_ids)
 if len(ids)!=n or pd.Series(ids).isna().any() or not pd.Index(ids).is_unique:raise ValueError('Kimlikler satırlarla eşleşmeli, benzersiz ve eksiksiz olmalı')
 rows=[dict(Item=item,Factor=f'F{np.flatnonzero(p[i])[0]+1}',b=float(params['difficulty'][i]),SharedStepGroup='GLOBAL',Loadings='FIXED UNIT',**{f'a{d+1}':int(p[i,d]) for d in range(q)}) for i,item in enumerate(x.columns)]
 steps=pd.DataFrame([dict(Step=k+1,From_category=labels[k],To_category=labels[k+1],Tau=float(t),Fixed=k==0,Estimated_globally=k!=0) for k,t in enumerate(params['thresholds'])])
 bounds=pd.DataFrame([dict(Parameter=name,Coordinate=i,Value=float(v),Lower=-6,Upper=6,At_bound=min(v+6,6-v)<=np.sqrt(np.finfo(float).eps)*max(1,abs(v))) for name,arr in params.items() for i,v in enumerate(arr) if name!='thresholds' or i!=0])
 metadata=dict(Model=outcome['model'],Status=STATUS,Engine='Python mirt '+mirt.__version__+' / RatingScaleModel',N=n,Items=j,Dimensions=q,Native_stacked_N=outcome['stacked_rows'],Observed_responses=n*j,Structural_missing_cells=n*(q-1)*j,
  Converged=bool(f.converged),Iterations=f.n_iterations,LogLik=f.log_likelihood,AIC=f.aic,BIC=-2*f.log_likelihood+f.n_parameters*np.log(n),Native_stacked_BIC=f.bic,
  BIC_definition='-2 joint logLik + P log(original N); native stacked BIC uses N*q and is retained separately',Parameters=f.n_parameters,Labels=labels,Shared_steps='ONE global vector; first step fixed zero',Covariance='Fixed identity, not estimated',
  Hessian='NOT COMPUTED; parameter SE/CI/Wald suppressed',Limitations=LIMITATION,**outcome['settings'])
 frames=dict(Parameters=pd.DataFrame(rows),SharedSteps=steps,Boundaries=bounds,Summary=pd.DataFrame([dict(Setting=k,Value=str(v)) for k,v in metadata.items()]),
  Fit=pd.DataFrame([dict(Factor='JOINT GLOBAL SHARED-STEP ML',Converged=f.converged,Iterations=f.n_iterations,LogLik=f.log_likelihood,AIC=f.aic,BIC=metadata['BIC'],Native_stacked_BIC=f.bic,Free_parameters=f.n_parameters)]),
  Pattern=pd.DataFrame(p.astype(int),index=x.columns,columns=[f'F{d+1}' for d in range(q)]).reset_index(names='Item'),Covariance=pd.DataFrame(np.eye(q),index=[f'F{d+1}' for d in range(q)],columns=[f'F{d+1}' for d in range(q)]).reset_index(names='Factor'),
  StructuralRows=pd.DataFrame([dict(Structural_row=d*n+i,Original_ID=ids[i],Factor=f'F{d+1}') for d in range(q) for i in range(n)]))
 if f.converged:
  for method in ('EAP','MAP'):
   scores=pd.DataFrame({'ID':ids})
   for d in range(q):
    responses=np.full(x.shape,-1,dtype=int);responses[:,p[:,d]]=x.to_numpy()[:,p[:,d]]
    native=mirt.fscores(f,responses,method=method,n_quadpts=outcome['settings']['n_quadpts'],person_ids=ids);theta=np.asarray(native.theta).reshape(-1)
    if len(theta)!=n or not np.isfinite(theta).all():raise ValueError('Geçersiz RSM kişi puanı')
    scores[f'F{d+1}']=theta
    if method=='EAP':
     sd=np.asarray(native.standard_error).reshape(-1)
     if not np.isfinite(sd).all() or np.any(sd<0):raise ValueError('Geçersiz RSM posterior SD')
     scores[f'Posterior_SD_F{d+1}']=sd
   frames[method]=scores
  frames.update(surface_tables(Curves(f.model,p,len(labels)),list(x.columns),labels,q))
 return dict(metadata=metadata,tables=frames)
