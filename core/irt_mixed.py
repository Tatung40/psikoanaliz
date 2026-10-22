"""Experimental B12.3 mixed 1PL-4PL marginal likelihood. Not connected to UI."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from core.irt_mg import Quadrature, _parts, as_matrix
from core.irt_mg34 import _single_thread
from core.irt_se_diagnostics import BOUNDARY_TOLERANCE
from core.rasch import SCORE_TOLERANCE

FREE = {"1PL":("b",),"2PL":("a","b"),"3PL":("a","b","c"),"4PL":("a","b","c","d")}

class MixedDesign:
    def __init__(self,item_types,groups=1,group_specific=(),a_upper=4.,d_lower=.65,prior=None):
        if not item_types or any(t not in FREE for t in item_types):
            raise ValueError("Madde türleri 1PL, 2PL, 3PL veya 4PL olmalıdır.")
        if groups not in (1,2): raise ValueError("Yalnızca bir veya iki grup desteklenir.")
        self.types,self.G=list(item_types),groups
        self.box={"a":(.15,float(a_upper)),"b":(-4.,4.),"c":(0.,.35),"d":(float(d_lower),1.)}
        if not (.15<float(a_upper) and 0.35<float(d_lower)<1.): raise ValueError("Parametre sınırları geçersiz.")
        k=len(self.types)
        if any(not isinstance(i,(int,np.integer)) or i<0 or i>=k for i in group_specific):
            raise ValueError("Gruba özgü madde indisi geçersiz.")
        self.idx={p:-np.ones((groups,k),int) for p in "abcd"}; self.labels=[]
        for i,t in enumerate(self.types):
            for p in FREE[t]:
                if i in group_specific and groups==2:
                    for g in range(2):
                        self.idx[p][g,i]=len(self.labels);self.labels.append((p,g,i))
                else:
                    self.idx[p][:,i]=len(self.labels);self.labels.append((p,-1,i))
        self.latent=len(self.labels);self.size=self.latent+(2 if groups==2 else 0)
        self.prior=dict(prior or {})
        for p,spec in self.prior.items():
            if p not in ("a","c","d") or len(spec)!=3 or spec[0] != ("lognormal" if p=="a" else "beta") or not all(np.isfinite(spec[1:])) or min(spec[1:])<=0 and p!="a" or (p=="a" and spec[2]<=0):
                raise ValueError("Önsel tanımı geçersiz: a lognormal, c/d beta.")
    def bounds(self):
        return [self.box[p] for p,_,_ in self.labels]+([(-3.,3.),(np.log(.2),np.log(5.))] if self.G==2 else [])
    def unpack(self,v,g):
        return [np.where(self.idx[p][g]>=0,v[np.maximum(self.idx[p][g],0)],fix) for p,fix in (("a",1.),("b",0.),("c",0.),("d",1.))]
    def latent_of(self,v,g):
        return (0.,1.) if g==0 else (float(v[self.latent]),float(np.exp(v[self.latent+1])))

def log_prior(v,design):
    val=0.;grad=np.zeros_like(v)
    for j,(p,_,_) in enumerate(design.labels):
        if p not in design.prior:continue
        kind,s1,s2=design.prior[p];x=v[j]
        if kind=="beta":
            xx=np.clip(x,1e-9,1-1e-9)
            val+=(s1-1)*np.log(xx)+(s2-1)*np.log1p(-xx)
            grad[j]+=(s1-1)/xx-(s2-1)/(1-xx)
        else:
            val+=-np.log(x)-(np.log(x)-s1)**2/(2*s2**2)
            grad[j]+=-1/x-(np.log(x)-s1)/(s2**2*x)
    return val,grad

def negloglik(v,design,parts,quad,n_total,with_prior=True):
    val=0.;grad=np.zeros_like(v)
    for g,(x1,x0) in enumerate(parts):
        mu,sd=design.latent_of(v,g);theta,lw=quad.nodes(mu,sd)
        a,b,c,d=design.unpack(v,g)
        L=expit(a*(theta[:,None]-b))
        P=np.clip(c+(d-c)*L,1e-12,1-1e-12)
        lp=x1@np.log(P).T+x0@np.log1p(-P).T+lw
        nc=logsumexp(lp,axis=1);val-=nc.sum();post=np.exp(lp-nc[:,None])
        G=(post.T@x1-(post.T@(x1+x0))*P)/(P*(1-P))
        GE=G*(d-c)*L*(1-L)
        terms={"a":(GE*(theta[:,None]-b)).sum(0),"b":-(GE*a).sum(0),
               "c":(G*(1-L)).sum(0),"d":(G*L).sum(0)}
        for p in "abcd":
            valid=design.idx[p][g]>=0
            np.add.at(grad,design.idx[p][g][valid],-terms[p][valid])
        if g==1:
            dt=(GE*a).sum(1)
            if quad.kind=="gauss-hermite":
                gmu,gls=dt.sum(),(dt*sd*quad.z).sum()
            else:
                r=post.sum(0);w=np.exp(lw);ng=r.sum()
                u=(theta-mu)/sd**2;q=(theta-mu)**2/sd**2
                gmu,gls=(r*u).sum()-ng*(w*u).sum(),(r*q).sum()-ng*(w*q).sum()
            grad[design.latent]-=gmu;grad[design.latent+1]-=gls
    if with_prior and design.prior:
        pv,pg=log_prior(v,design);val-=pv;grad-=pg
    return val/n_total,grad/n_total

def start_values(x,obs,design,rng=None,jitter=0.):
    p=np.clip(x.sum(0)/np.maximum(obs.sum(0),1),.05,.95)
    base={"a":1.,"c":.1,"d":.95}
    lo,hi=np.array(design.bounds()).T
    v=np.array([(-np.log(p[i]/(1-p[i])) if q=="b" else base[q]) for q,_,i in design.labels]+[0.]*(design.size-design.latent))
    if rng is not None and jitter:
        scales={"a":.5,"b":.5,"c":.1,"d":.1}
        v[:design.latent]+=rng.normal(0,jitter,design.latent)*np.array([scales[q] for q,_,_ in design.labels])
    return np.clip(v,lo+1e-3,hi-1e-3)

def fit_mixed(data,item_types,groups=None,reference=None,focal=None,group_specific=(),
              a_upper=4.,d_lower=.65,prior=None,quadrature="mirt",points=61,
              starts=1,seed=2026,max_iter=20000):
    x,obs=as_matrix(data)
    names=list(data.columns) if isinstance(data,pd.DataFrame) else [f"M{i+1}" for i in range(x.shape[1])]
    if len(item_types)!=x.shape[1]:raise ValueError("Her madde için model türü gereklidir.")
    if len(set(names))!=len(names):raise ValueError("Madde isimleri benzersiz olmalıdır.")
    if not isinstance(starts,(int,np.integer)) or not 1<=starts<=20 or max_iter<1:
        raise ValueError("Başlangıç ve yineleme sayısı geçersiz.")
    if quadrature=="mirt" and points!=61:
        raise ValueError("mirt kuadratürü sabit 61 düğüm kullanır; points=61 olmalıdır.")
    if groups is None:
        if group_specific:raise ValueError("Tek grupta DIF maddesi belirtilemez.")
        gv=np.zeros(len(x),int);G=1
    else:
        gs=pd.Series(groups).reset_index(drop=True)
        if len(gs)!=len(x) or reference is None or focal is None or reference==focal or gs.isna().any():
            raise ValueError("Grup uzunluğu, referans ve odak grupları geçerli olmalıdır.")
        if not gs.isin([reference,focal]).all():raise ValueError("Tanımsız grup düzeyleri var; gözlemler sessizce çıkarılamaz.")
        gv=(gs==focal).to_numpy(dtype=int);G=2
        if np.sum(gv==0)<30 or np.sum(gv==1)<30:raise ValueError("Her grupta en az 30 kişi gerekir.")
    unknown=set(group_specific)-set(names)
    if unknown:raise ValueError("Bilinmeyen DIF maddeleri: "+str(sorted(unknown)))
    idx=[names.index(n) for n in group_specific]
    if G==2 and len(idx)==len(names):raise ValueError("En az bir çapa madde gereklidir.")
    design=MixedDesign(item_types,G,idx,a_upper,d_lower,prior)
    quad=Quadrature(quadrature,points);parts=_parts(x,obs,gv,G);n=len(x)
    bounds=design.bounds();lo,hi=np.array(bounds).T;rng=np.random.default_rng(seed);runs=[]
    with _single_thread():
        for s in range(starts):
            v0=start_values(x,obs,design,rng,1. if s else 0.)
            res=minimize(negloglik,v0,args=(design,parts,quad,n),jac=True,method="L-BFGS-B",bounds=bounds,
                         options={"maxiter":max_iter,"maxfun":4*max_iter,"ftol":1e-15,"gtol":1e-12,"maxcor":30})
            v=np.asarray(res.x,float);f,g=negloglik(v,design,parts,quad,n)
            pg=g.copy();low=v<=lo+1e-8;up=v>=hi-1e-8
            pg[low]=np.minimum(g[low],0);pg[up]=np.maximum(g[up],0)
            if np.isfinite(f) and np.isfinite(pg).all():
                runs.append(dict(v=v,objective=-f*n,kkt=float(np.abs(pg).max()),iterations=int(res.nit)))
    if not runs:raise RuntimeError("Hiçbir başlangıç geçerli kestirim üretmedi.")
    good=[r for r in runs if r["kkt"]<=SCORE_TOLERANCE]
    best=max(good or runs,key=lambda r:r["objective"])
    v=best["v"];near=(v-lo<=BOUNDARY_TOLERANCE)|(hi-v<=BOUNDARY_TOLERANCE);near[design.latent:]=False
    ll=-negloglik(v,design,parts,quad,n,with_prior=False)[0]*n
    tables=[]
    for g in range(G):
        a,b,c,d=design.unpack(v,g)
        tables.append(pd.DataFrame({"Madde":names,"tur":design.types,"a":a,"b":b,"c":c,"d":d}))
    close=[r for r in runs if best["objective"]-r["objective"]<.01]
    return {"items":tables[0],"items_focal":tables[1] if G==2 else None,"loglik":ll,
            "objective":best["objective"],"map":bool(design.prior),"kkt_per_person":best["kkt"],
            "converged":best["kkt"]<=SCORE_TOLERANCE,
            "bound":[(design.labels[j][0],names[design.labels[j][2]],design.labels[j][1]) for j in np.where(near)[0]],
            "latent_focal":design.latent_of(v,1) if G==2 else None,"params":v,"design":design,
            "starts":{"n":len(runs),"converged":len(good),
                      "objective_range":max(r["objective"] for r in runs)-min(r["objective"] for r in runs),
                      "within_0.01_of_best":len(close),
                      "max_param_spread_among_them":float(max(np.abs(r["v"]-v).max() for r in close)) if close else np.nan},
            "n":n,"free_parameters":design.size,"quadrature":quadrature}

def loglik_at(data,item_types,params_table,quadrature="mirt",points=61):
    x,obs=as_matrix(data);design=MixedDesign(item_types)
    t=params_table.reset_index(drop=True);v=np.array([t.loc[i,p] for p,_,i in design.labels])
    return -negloglik(v,design,_parts(x,obs,np.zeros(len(x),int),1),Quadrature(quadrature,points),len(x))[0]*len(x)
