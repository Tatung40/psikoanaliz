"""Explicit SciPy route; existing Python mirt estimators remain unchanged."""
import numpy as np
import pandas as pd
import scipy
from core import irt_multidim_general as engine
from core.irt_multidim import STATUS,SURFACE_POINTS,surface_tables

CONFIG={
 '3PL — genel pozitif CFA, c sabit (SciPy)':('3PL',False,False),
 '4PL — genel pozitif CFA, c/d sabit (SciPy)':('4PL',False,False),
 '3PL — 2D between-item, c serbest (SciPy)':('3PL',True,False),
 '4PL — 2D between-item, c/d serbest (SciPy)':('4PL',True,True),
}
MODELS=tuple(CONFIG)

def allowed_dimensions(model):return [2] if CONFIG[model][1] else [2,3]
LIMITATION=('EXPERIMENTAL / NOT CERTIFIED. Ayrı SciPy marjinal ML bileşeni; Python mirt 1.2.0 genel 3PL/4PL API desteği değildir. '
 'Yalnız 2D/3D, pozitif CFA, boyut başına en az üç saf çapa, ortalama0/birim varyans; tam korelasyon matrisi kestirilir. '
 'Model adındaki sabit c/d biliniyor varsayılır; serbest c/d yalnız doğrulanmış 2D between-item rotalarındadır. Mevcut bağımsız R kabulü GH15 sayısal karşılaştırmasıdır; popülasyon geri kazanımı/Tip-I hata/GA kapsaması değildir. '
 'SE/GA/p, LR ve doğrulanmamış MAP kapalıdır. EAP SD/tam posterior kovaryans ve Fisher SEM madde parametrelerine koşulludur. '
 'Sonlu GH önseli normalize edilir; quadrature doğruluğu veya küresel optimum garanti edilmez.')


def specification(data,model,pattern,correlated='full',exploratory=False,c=.2,u=1.):
 if model not in MODELS:raise ValueError('Bu genel aile bağımsız doğrulama kapısından geçmedi')
 if correlated!='full' or exploratory:raise ValueError('Genel rota yalnız tam ilişkili, tanımlanmış pozitif CFA; diğer yolları mevcut motorla seçin')
 x=pd.DataFrame(data)
 family,free_c,free_u=CONFIG[model];spec=engine.Specification(np.asarray(pattern),family,True,free_c,free_u,c=c,u=u)
 if free_c and (spec.q!=2 or not np.all(spec.pattern.sum(1)==1)):raise ValueError('Serbest c/d yalnız bağımsız doğrulanmış 2D between-item profilinde; 3D R doğrulaması BLOCKED')
 if len(x.columns)!=spec.j or not x.columns.is_unique:raise ValueError('Madde/Q-matrix eşleşmesi veya benzersiz adlar geçersiz')
 if SURFACE_POINTS**2*spec.j*spec.q**2>250_000:raise ValueError('Rapor bellek bütçesi: en fazla 250000 bilgi matrisi satırı')
 return x,spec


class Curves:
 def __init__(self,spec,values):self.spec=spec;self.values=values
 def probability(self,theta):return engine.probabilities(self.spec,self.values,theta)
 def item_information_matrix(self,theta,item_idx):return engine.information(self.spec,self.values,theta)[:,item_idx]
 def test_information_matrix(self,theta):return engine.information(self.spec,self.values,theta).sum(1)


def fit_report(data,model,pattern,*,correlated='full',exploratory=False,c=.2,u=1.,n_quadpts=15,max_iter=2000,tol=1e-13,person_ids=None):
 x,spec=specification(data,model,pattern,correlated,exploratory,c,u);ids=list(x.index) if person_ids is None else list(person_ids)
 if len(ids)!=len(x) or pd.Series(ids).isna().any() or not pd.Index(ids).is_unique:raise ValueError('Kimlikler benzersiz, eksiksiz ve satırlarla eşleşmeli')
 native=engine.fit(x,spec,n_quadpts=n_quadpts,max_iter=max_iter,tol=tol);values=native['parameters'];A,delta,lower,upper,eta=spec.unpack(values);C=spec.covariance(eta)
 params=pd.DataFrame([dict(Item=item,delta=delta[i],c=lower[i],d=upper[i],**{f'a{d+1}':A[i,d] for d in range(spec.q)}) for i,item in enumerate(x)])
 _,bounds=spec.start_bounds();names=[f'a[{list(x)[i]},F{d+1}]' for i,d in zip(*np.where(spec.pattern))]+[f'delta[{item}]' for item in x]+([f'c[{item}]' for item in x] if spec.free_c else [])+([f'd[{item}]' for item in x] if spec.free_u else [])+[f'eta{r+1}{s+1}' for r,s in zip(*spec.pairs)]
 diagnostics=native['diagnostics'];active=set(diagnostics['active_bounds']);boundary=pd.DataFrame([dict(Parameter=name,Value=v,Lower=b[0],Upper=b[1],At_bound=k in active) for k,(name,v,b) in enumerate(zip(names,values,bounds))])
 metadata=dict(Model=model,Status=STATUS,Backend='SCIPY_GENERAL',Engine='SciPy '+scipy.__version__+' direct fixed-grid marginal ML (not Python mirt calibration)',N=len(x),Items=spec.j,Dimensions=spec.q,Labels=('0','1'),
  Converged=native['converged'],Iterations=native['iterations'],LogLik=native['logLik'],AIC=native['AIC'],BIC=native['BIC'],Parameters=spec.n_parameters,
  Fixed_c=not spec.free_c,Fixed_d=not spec.free_u,n_quadpts=n_quadpts,max_iter=max_iter,tol=tol,Identification='Named positive pure anchors; mean0/unit variances; covariance normalized-Cholesky eta [-3,3]',
  Optimizer_stop=diagnostics['message'],Projected_score_max=diagnostics['projected_score_max'],At_box_bound=bool(active),
  Hessian='Not computed in interactive calibration; parameter SE/CI/LR disabled; independent benchmark tangent Hessians in validation evidence',MAP='NOT VALIDATED / DISABLED',Limitations=LIMITATION)
 frames=dict(Parameters=params,Boundaries=boundary,Fit=pd.DataFrame([dict(Converged=native['converged'],Iterations=native['iterations'],LogLik=native['logLik'],AIC=native['AIC'],BIC=native['BIC'],Free_parameters=spec.n_parameters,Projected_score_max=diagnostics['projected_score_max'])]),
  Covariance=pd.DataFrame(C,index=[f'F{d+1}' for d in range(spec.q)],columns=[f'F{d+1}' for d in range(spec.q)]).reset_index(names='Factor'),
  Pattern=pd.DataFrame(spec.pattern.astype(int),index=x.columns,columns=[f'F{d+1}' for d in range(spec.q)]).reset_index(names='Item'),
  OptimizerTrace=pd.DataFrame(diagnostics['trace']),Scores=pd.DataFrame(dict(Parameter=names,Physical_gradient=diagnostics['gradient'])))
 if native['converged']:
  mean,cov=engine.eap(native['objective'],values)
  if not np.isfinite(mean).all() or not np.isfinite(cov).all() or np.any(np.diagonal(cov,axis1=1,axis2=2)<0):raise ValueError('Geçersiz posterior momentler')
  scores=pd.DataFrame(mean,columns=[f'F{d+1}' for d in range(spec.q)]);scores.insert(0,'ID',ids)
  for d in range(spec.q):scores[f'Posterior_SD_F{d+1}']=np.sqrt(cov[:,d,d])
  frames['EAP']=scores;frames['PosteriorCovariance']=pd.DataFrame([dict(ID=ids[n],Row=r+1,Column=s+1,Covariance=cov[n,r,s]) for n in range(len(x)) for r in range(spec.q) for s in range(spec.q)])
  frames.update(surface_tables(Curves(spec,values),list(x),metadata['Labels'],spec.q))
 frames['Summary']=pd.DataFrame([dict(Setting=k,Value=str(v)) for k,v in metadata.items()])
 return dict(metadata=metadata,tables=frames)
