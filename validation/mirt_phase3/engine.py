"""Experimental SciPy marginal ML prototype, NOT Python mirt's unsupported API.

Frozen GH support and normalized Gaussian density ratio match the established
R customGroup integration. No SE/CI/LR inference; no production activation.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit,logsumexp,log_expit
from scipy.optimize import minimize
from mirt.estimation.quadrature import GaussHermiteQuadrature
from core.irt_multidim import structure


@dataclass
class Specification:
 pattern: np.ndarray
 family: str='3PL'
 correlated: bool=True
 free_c: bool=False
 free_u: bool=False
 c: float=.2
 u: float=1.

 def __post_init__(self):
  self.pattern=np.asarray(self.pattern,bool);self.q=self.pattern.shape[1] if self.pattern.ndim==2 else 0
  if self.q not in (2,3):raise ValueError('Phase3 only 2D/3D; 4D/5D NOT CERTIFIED')
  structure(self.pattern,False,False)
  if self.family not in ('2PL','3PL','4PL'):raise ValueError('Unsupported family')
  if self.family=='2PL':self.c=0.;self.u=1.
  if self.family=='3PL':self.u=1.
  if (self.free_u and self.family!='4PL') or (self.free_c and self.family=='2PL'):raise ValueError('Invalid free asymptote')
  if not (0<=self.c<=.5 and .5<=self.u<=1 and self.c<self.u):raise ValueError('0<=c<d<=1 required')
  self.j=len(self.pattern);self.active=np.flatnonzero(self.pattern);self.pairs=np.tril_indices(self.q,-1)
  self.n_cov=len(self.pairs[0]) if self.correlated else 0
  self.n_parameters=len(self.active)+self.j+self.j*self.free_c+self.j*self.free_u+self.n_cov

 def start_bounds(self):
  start=list(np.full(len(self.active),1.2))+[0.]*self.j;bounds=[(.1,5.)]*len(self.active)+[(-6.,6.)]*self.j
  if self.free_c:start+=list(np.full(self.j,self.c));bounds +=[(0.,.5)]*self.j
  if self.free_u:start+=list(np.full(self.j,self.u));bounds +=[(.5,1.)]*self.j
  if self.correlated:start +=[0.]*self.n_cov;bounds +=[(-3.,3.)]*self.n_cov
  return np.array(start),bounds

 def unpack(self,x):
  n=len(self.active);A=np.zeros_like(self.pattern,float);A.flat[self.active]=x[:n];delta=x[n:n+self.j];offset=n+self.j
  c=np.full(self.j,self.c);u=np.full(self.j,self.u)
  if self.free_c:c=x[offset:offset+self.j];offset+=self.j
  if self.free_u:u=x[offset:offset+self.j];offset+=self.j
  return A,delta,c,u,x[offset:]

 def covariance(self,eta,derivatives=False):
  L=np.eye(self.q)
  if self.correlated:L[self.pairs]=eta
  S=L@L.T;v=np.diag(S);scale=np.sqrt(np.outer(v,v));C=S/scale
  if not derivatives:return C
  changes=[]
  for r,c in zip(*self.pairs) if self.correlated else []:
   E=np.zeros_like(L);E[r,c]=1;dS=E@L.T+L@E.T;dv=np.diag(dS)
   changes.append(dS/scale-C*(dv[:,None]/v[:,None]+dv[None,:]/v[None,:])/2)
  return C,changes


def log_probabilities(spec,x,theta):
 A,d,c,u,_=spec.unpack(x);z=np.asarray(theta)@A.T+d
 if np.any(c>=u) or np.any(c<0) or np.any(u>1):raise ValueError('Invalid physical asymptotes')
 logc=np.log(c,out=np.full_like(c,-np.inf),where=c>0)
 log1u=np.log(1-u,out=np.full_like(u,-np.inf),where=u<1)
 width=np.log(u-c);ls=log_expit(z);lq=log_expit(-z)
 return np.logaddexp(logc,width+ls),np.logaddexp(log1u,width+lq),ls,lq


def probabilities(spec,x,theta):
 return np.exp(log_probabilities(spec,x,theta)[0])


def information(spec,x,theta):
 A,d,c,u,_=spec.unpack(x);lp,lq,ls,lt=log_probabilities(spec,x,theta)
 scale=np.exp(2*(np.log(u-c)+ls+lt)-lp-lq)
 return np.einsum('jr,js,nj->njrs',A,A,scale)


class Objective:
 def __init__(self,data,spec,n_quadpts=15):
  data=np.asarray(data)
  if data.ndim!=2 or data.shape[1]!=spec.j or len(data)<30 or not np.isin(data,[0,1]).all() or any(np.unique(data[:,j]).size!=2 for j in range(spec.j)):raise ValueError('Complete binary data, N>=30 and both categories per item required')
  if n_quadpts<5 or n_quadpts**spec.q>4000 or len(data)*spec.j*n_quadpts**spec.q>20_000_000:raise ValueError('GH computation budget exceeded')
  self.data=data.astype(float);self.spec=spec;self.grid=GaussHermiteQuadrature(n_quadpts,spec.q);self.calls=0

 def masses(self,eta,gradient=False,log=False):
  C,derivatives=self.spec.covariance(eta,True);inverse=np.linalg.inv(C);theta=self.grid.nodes
  sign,ld=np.linalg.slogdet(C)
  if sign<=0:raise ValueError('Nonpositive covariance')
  logw=np.log(self.grid.weights)-ld/2-np.einsum('kr,rs,ks->k',theta,inverse-np.eye(self.spec.q),theta)/2
  logw-=logsumexp(logw);w=np.exp(logw)
  if not gradient:return logw if log else w
  transformed=theta@inverse
  scores=np.column_stack([-.5*np.trace(inverse@d)+.5*np.einsum('kr,rs,ks->k',transformed,d,transformed) for d in derivatives]) if derivatives else np.empty((len(theta),0))
  scores-=w@scores
  return (logw if log else w),scores

 def evaluate(self,x,posterior=False):
  self.calls+=1;s=self.spec;A,d,c,u,eta=s.unpack(x)
  if np.any(c>=u):raise ValueError('Asymptotes overlap; invalid parameter space')
  lp,lq,ls,lt=log_probabilities(s,x,self.grid.nodes)
  logw,dw=self.masses(eta,True,log=True);ell=self.data@lp.T+(1-self.data)@lq.T+logw
  marginal=logsumexp(ell,axis=1);post=np.exp(ell-marginal[:,None]);ll=float(marginal.sum())
  counts=post.T@self.data;failures=post.T@(1-self.data)
  ld=np.log(u-c)+ls+lt
  slope_score=counts*np.exp(ld-lp)-failures*np.exp(ld-lq);gA=slope_score.T@self.grid.nodes;gd=slope_score.sum(0)
  parts=[gA.ravel()[s.active],gd]
  if s.free_c:parts.append((counts*np.exp(lt-lp)-failures*np.exp(lt-lq)).sum(0))
  if s.free_u:parts.append((counts*np.exp(ls-lp)-failures*np.exp(ls-lq)).sum(0))
  if s.correlated:parts.append(post.sum(0)@dw)
  result=(-ll,-np.concatenate(parts))
  return (*result,post) if posterior else result


def fit(data,spec,*,n_quadpts=15,max_iter=2000,tol=1e-13,start=None):
 objective=Objective(data,spec,n_quadpts);x0,bounds=spec.start_bounds();x0=x0 if start is None else np.asarray(start,float)
 if len(x0)!=spec.n_parameters or any(v<lo or v>hi for v,(lo,hi) in zip(x0,bounds)):raise ValueError('Invalid start')
 trace=[]
 def callback(x):
  value,gradient=objective.evaluate(x);trace.append(dict(logLik=-value,max_score=float(np.max(abs(gradient)))))
 result=minimize(objective.evaluate,x0,jac=True,method='L-BFGS-B',bounds=bounds,callback=callback,
  options=dict(maxiter=max_iter,ftol=tol,gtol=1e-7,maxls=40))
 value,gradient=objective.evaluate(result.x);B=np.array(bounds);active=(result.x-B[:,0]<=np.sqrt(np.finfo(float).eps)*np.maximum(1,abs(result.x)))|(B[:,1]-result.x<=np.sqrt(np.finfo(float).eps)*np.maximum(1,abs(result.x)))
 # Projected scores distinguish legitimate box KKT conditions from flattening.
 projected=gradient.copy();projected[(result.x<=B[:,0])&(gradient>0)]=0;projected[(result.x>=B[:,1])&(gradient<0)]=0
 return dict(parameters=result.x,specification=spec,objective=objective,converged=bool(result.success),iterations=result.nit,
  logLik=-value,AIC=2*spec.n_parameters+2*value,BIC=spec.n_parameters*np.log(len(data))+2*value,
  diagnostics=dict(engine='SciPy direct fixed-grid marginal ML; NOT Python mirt calibration',message=str(result.message),objective_calls=objective.calls,
   gradient=gradient.tolist(),projected_score_max=float(np.max(abs(projected))),active_bounds=np.flatnonzero(active).tolist(),trace=trace,
   Hessian='NOT COMPUTED; no SE/CI/LR; local numerical convergence only',certification='NOT CERTIFIED'))


def eap(objective,x):
 _,_,post=objective.evaluate(x,True);theta=objective.grid.nodes;mean=post@theta
 cov=np.einsum('nk,kr,ks->nrs',post,theta,theta)-np.einsum('nr,ns->nrs',mean,mean)
 return mean,cov
