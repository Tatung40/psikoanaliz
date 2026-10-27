"""Exact orthogonal between-item composition of public 1D Python mirt fits.

No new psychometric estimator, model equation, or derivative implementation.
"""
import platform
import numpy as np
import pandas as pd
import mirt
from core.irt_multidim import STATUS,surface_tables
from core.irt_python_mirt_adapter import normalize_solver_settings

MODELS=('3PL — bağımsız bloklar, c sabit','4PL — bağımsız bloklar, c ve d sabit')
LIMITATION=('EXPERIMENTAL / NOT CERTIFIED. Yalnız between-item, ortogonal, bağımsız N(0,1) boyutlar. '
 'Her madde tek boyuta yüklenir; pozitif a, sabit c (ve 4PL için sabit üst asimptot). '
 'Ortak olabilirlik bağımsız blok olabilirliklerinin çarpımıdır; ayrı library fitleri tam ortak ML verir. '
 'Within-item, ilişkili boyutlar ve serbest/ortak kestirilen asimptotlar kapalıdır. '
 'SE/GA/p, LR, Hessian temelli çıkarım ve doğrulanmamış uyum istatistikleri yoktur. '
 'Sabit asimptotlar biliniyor varsayılır; kestirim belirsizliğine katılmaz. '
 'Yakınsama küresel optimum veya bilimsel sertifikasyon değildir.')


def validate(data,pattern,correlated=False,exploratory=False,c=.2,u=1.):
 x=pd.DataFrame(data);p=np.asarray(pattern)
 if p.ndim!=2 or p.shape[1] not in (2,3,4,5) or p.shape[0]!=len(x.columns) or not np.isin(p,[0,1]).all():raise ValueError('2–5 boyutlu geçerli madde matrisi gerekli')
 if correlated or exploratory or not np.all(p.sum(1)==1):raise ValueError('Yalnız ortogonal between-item yapı: madde başına tam bir yük')
 if any(p[:,j].sum()<6 for j in range(p.shape[1])):raise ValueError('Boyut başına en az 6 madde gerekli; bu bilimsel yeterlilik sertifikası değildir')
 if len(x)<30 or x.isna().any().any() or not x.columns.is_unique or not x.isin([0,1]).all().all():raise ValueError('En az 30 kişi, benzersiz maddeler, eksiksiz 0/1 veri gerekli')
 if any(x[col].nunique()!=2 for col in x):raise ValueError('Her maddede 0 ve 1 gözlenmeli')
 if not (np.isfinite(c) and np.isfinite(u) and 0<=c<=.5 and .5<=u<=1 and c<u):raise ValueError('0≤c≤.5, .5≤u≤1 ve c<u gerekli')
 return x.astype(int),p.astype(bool)


class Curves:
 """Postfit dispatch only; never impersonates a library FitResult/model."""
 def __init__(self,blocks,pattern):self.blocks=blocks;self.pattern=np.asarray(pattern,bool)
 def probability(self,theta):
  theta=np.asarray(theta);out=np.empty((len(theta),len(self.pattern)))
  for d,index,fit in self.blocks:out[:,index]=fit.model.probability(theta[:,d,None])
  return out
 def item_information_matrix(self,theta,item_idx):
  q=self.pattern.shape[1];d=int(np.flatnonzero(self.pattern[item_idx])[0]);out=np.zeros((len(theta),q,q))
  _,index,f=next(block for block in self.blocks if block[0]==d);local=int(np.flatnonzero(index==item_idx)[0])
  out[:,d,d]=np.asarray(f.model.information(np.asarray(theta)[:,d,None],item_idx=local)).reshape(-1)
  return out
 def test_information_matrix(self,theta):
  out=np.zeros((len(theta),self.pattern.shape[1],self.pattern.shape[1]))
  for d,index,f in self.blocks:out[:,d,d]=np.asarray(f.model.information(np.asarray(theta)[:,d,None])).sum(axis=1)
  return out


def fit(data,model,pattern,*,correlated=False,exploratory=False,c=.2,u=1.,n_quadpts=61,max_iter=1000,tol=1e-8):
 if model not in MODELS:raise ValueError('Doğrulanmamış faktörleştirilmiş aile')
 family='3PL' if model.startswith('3PL') else '4PL'
 if family=='3PL' and u!=1:raise ValueError('3PL üst asimptotu 1 olmalı')
 x,p=validate(data,pattern,correlated,exploratory,c,u);settings=normalize_solver_settings(max_iter=max_iter,tol=tol,n_quadpts=n_quadpts)
 if n_quadpts<5 or n_quadpts>81 or len(x)*len(x.columns)*n_quadpts>20_000_000:raise ValueError('Bağımsız blok GH/bellek bütçesi aşıldı')
 blocks=[]
 for d in range(p.shape[1]):
  index=np.flatnonzero(p[:,d]);n=len(index);start=dict(discrimination=np.ones(n),difficulty=np.zeros(n),guessing=np.full(n,c));fixed=dict(guessing=True)
  if family=='4PL':start['upper']=np.full(n,u);fixed['upper']=True
  result=mirt.fit_mirt(x.iloc[:,index],model=family,n_factors=1,start_values=start,fixed=fixed,priors=None,compute_standard_errors=False,accelerate='none',**settings)
  if not np.array_equal(result.model.parameters['guessing'],np.full(n,c)) or family=='4PL' and not np.array_equal(result.model.parameters['upper'],np.full(n,u)):raise ValueError('Sabit asimptot değişti')
  blocks.append((d,index,result))
 return dict(model=model,family=family,blocks=blocks,pattern=p,coded=x,settings=settings,c=c,u=u,labels=('0','1'))


def report(outcome,person_ids=None):
 blocks=outcome['blocks'];x=outcome['coded'];p=outcome['pattern'];q=p.shape[1];frames={};rows=[];bounds=[];fit_rows=[]
 for d,index,f in blocks:
  values=f.model.parameters
  fit_rows.append(dict(Factor=f'F{d+1}',Converged=bool(f.converged),Iterations=f.n_iterations,LogLik=f.log_likelihood,AIC=f.aic,BIC=f.bic,Free_parameters=f.n_parameters))
  for local,item in enumerate(index):
   a=float(values['discrimination'][local]);b=float(values['difficulty'][local])
   rows.append(dict(Item=x.columns[item],Factor=f'F{d+1}',b=b,Intercept=-a*b,c=outcome['c'],d=outcome['u'],Asymptotes='FIXED / NOT ESTIMATED',**{f'a{j+1}':a if j==d else 0. for j in range(q)}))
   for name,val,lo,hi in [('a',a,.1,5),('b',b,-6,6)]:bounds.append(dict(Item=x.columns[item],Parameter=name,Value=val,Lower=lo,Upper=hi,At_bound=min(val-lo,hi-val)<=np.sqrt(np.finfo(float).eps)*max(1,abs(val))))
 fit_frame=pd.DataFrame(fit_rows);ll=float(fit_frame.LogLik.sum());npar=int(fit_frame.Free_parameters.sum())
 metadata=dict(Model=outcome['model'],Status=STATUS,Engine='Python mirt '+mirt.__version__,Python=platform.python_version(),N=len(x),Items=len(x.columns),Dimensions=q,
  Converged=bool(fit_frame.Converged.all()),LogLik=ll,AIC=2*npar-2*ll,BIC=np.log(len(x))*npar-2*ll,Parameters=npar,Labels=outcome['labels'],
  Likelihood='Exact product of independent orthogonal between-item marginal likelihoods',Covariance='Fixed identity; not estimated',
  Estimation='Marginal ML; no item priors',Hessian='NOT COMPUTED; SE/CI/Wald suppressed',MAP_diagnostics='Public scalar API point estimates; optimizer status unexposed; no MAP SE',Fixed_c=outcome['c'],Fixed_u=outcome['u'],Limitations=LIMITATION,**outcome['settings'])
 frames.update(Parameters=pd.DataFrame(rows),Fit=fit_frame,Boundaries=pd.DataFrame(bounds),
  Pattern=pd.DataFrame(p.astype(int),index=x.columns,columns=[f'F{j+1}' for j in range(q)]).reset_index(names='Item'),
  Covariance=pd.DataFrame(np.eye(q),index=[f'F{j+1}' for j in range(q)],columns=[f'F{j+1}' for j in range(q)]).reset_index(names='Factor'),
  Summary=pd.DataFrame([dict(Setting=k,Value=str(v)) for k,v in metadata.items()]))
 if metadata['Converged']:
  ids=list(x.index) if person_ids is None else list(person_ids)
  if len(ids)!=len(x) or pd.Series(ids).isna().any() or not pd.Index(ids).is_unique:raise ValueError('Kimlikler benzersiz, eksiksiz ve satırlarla eşleşmeli')
  for method in ('EAP','MAP'):
   scores=pd.DataFrame({'ID':ids})
   for d,index,f in blocks:
    native=mirt.fscores(f,x.iloc[:,index].to_numpy(),method=method,n_quadpts=outcome['settings']['n_quadpts'],person_ids=ids)
    values=np.asarray(native.theta).reshape(-1)
    if len(values)!=len(x) or not np.isfinite(values).all():raise ValueError('Geçersiz kişi puanı; raporlama engellendi')
    scores[f'F{d+1}']=values
    if method=='EAP':
     sd=np.asarray(native.standard_error).reshape(-1)
     if len(sd)!=len(x) or not np.isfinite(sd).all() or np.any(sd<0):raise ValueError('Geçersiz posterior SD; raporlama engellendi')
     scores[f'Posterior_SD_F{d+1}']=sd
   frames[method]=scores
  frames.update(surface_tables(Curves(blocks,p),list(x.columns),outcome['labels'],q))
 return dict(metadata=metadata,tables=frames)
