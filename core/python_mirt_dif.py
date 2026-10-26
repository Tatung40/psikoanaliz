"""Public Python mirt multigroup ML fits; no local psychometric estimator.

Explicit-anchor add design: H1 frees the selected families on every nonanchor;
H0 constrains one studied item. Retain both fits (unlike multigroup_dif's table).
No Wald or inferential release without multigroup regularity validation.
"""
import hashlib
import inspect
import platform
import numpy as np
import pandas as pd
from core.irt_b14 import prepare_polytomous
from core.irt_python_mirt_adapter import normalize_solver_settings
from reports.irt_b14_report import category_frequencies

MODELS=('Rasch','1PL','2PL','3PL','4PL','GRM','PCM','GPCM','NRM')
FAMILIES={'Uniform / konum':('intercepts',),'Nonuniform / eğim':('discrimination',),
          'Omnibus':('discrimination','intercepts')}
LIMITATION=('NOT CERTIFIED: bağımsız R çok gruplu DIF doğrulaması ve tasarıma özgü Tip-I/FDR '
            'kalibrasyonu yok. Çok gruplu Hessian/SE tanısı public API tarafından üretilmiyor; '
            'Optimizasyon sınırı tanısı yok; global optimum veya tam tanımlanabilirlik iddiası yok. '
            'p/q, karar ve Wald kapalıdır. LR, eğriler ve parametre farkları betimseldir. '
            'Çapaların değişmezliği varsayılır; otomatik çapa seçimi yapılmaz. '
            'NRM kategori kodları başarı sırası değildir. Eğim testi tek başına saf nonuniform '
            'DIF kanıtı değildir; uniform test eşit eğim varsayar. 3PL/4PL c/u gruplarda eşittir; '
            'önceki bootstrap çıkarım kapıları aynen korunur.')


def support_matrix():
    rows=[]
    for model in MODELS+('RSM',):
        fixed=model in ('Rasch','1PL','PCM')
        rows.append(dict(Model=model,Calibration=model!='RSM',LR=model!='RSM',Wald=False,
            Slope_test=not fixed and model!='RSM',Inferential_p=False,
            Note=('Public multigroup model factory RSM desteklemiyor' if model=='RSM' else
                  'Public multigroup 1PL a=1; ortak serbest a desteklenmiyor' if model in ('Rasch','1PL') else
                  'PCM a=1' if model=='PCM' else 'c/u DIF testi yok' if model in ('3PL','4PL') else
                  'NRM kategori0 A=D=0; nominal kontrastlar' if model=='NRM' else 'Konum/eğim aileleri')))
    return pd.DataFrame(rows)


def fingerprint(data,settings):
    frame=pd.DataFrame(data)
    payload=pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes()
    schema=(tuple(frame.columns),tuple(map(str,frame.dtypes)),frame.shape,settings,'mg-dif-v1')
    return hashlib.sha256(payload+repr(schema).encode()).hexdigest()


def prepare(data,items,group,reference,focal,model,labels=None):
    if model not in MODELS:raise ValueError('Desteklenmeyen çok gruplu model')
    if not data.columns.is_unique or any(not isinstance(c,str) for c in data.columns):
        raise ValueError('Sütun adları benzersiz metin olmalı')
    if len(items)<3 or len(set(items))!=len(items) or group in items:
        raise ValueError('En az üç benzersiz madde; grup maddelerden ayrı olmalı')
    if reference==focal or pd.isna(reference) or pd.isna(focal):raise ValueError('İki ayrı grup gerekli')
    if data[group].isna().any():raise ValueError('Eksik grup etiketi; otomatik satır silinmedi')
    selected=data.loc[data[group].isin([reference,focal])]
    g=np.where(selected[group].eq(reference),'reference','focal')
    if set(g)!=set(('reference','focal')):raise ValueError('İki seçili grup veride bulunmalı')
    if model in ('Rasch','1PL','2PL','3PL','4PL'):
        x=selected[items].apply(pd.to_numeric,errors='raise');lab=('0','1')
        if not np.isfinite(x.to_numpy(float)).all() or not x.isin([0,1]).all().all():
            raise ValueError('Eksiksiz 0/1 madde yanıtları gerekli')
        x=x.astype(int)
    else:x,lab=prepare_polytomous(selected[items],model,categories=labels if model=='NRM' else None)
    frames=[]
    for name in ('reference','focal'):
        sub=x.loc[g==name]
        if len(sub)<=len(items):raise ValueError('Her grubun N değeri madde sayısından büyük olmalı; bu yeterli örneklem güvencesi değildir')
        if any(set(sub[c])!=set(range(len(lab))) for c in items):
            raise ValueError('Her grupta her maddede tüm kategoriler gözlenmeli; kategori uydurulmaz')
        freq=category_frequencies(sub,lab);freq.insert(0,'Grup',name);frames.append(freq)
    return x,g,lab,pd.concat(frames,ignore_index=True),len(data)-len(selected)


def gate(constrained,free,model):
    """Never clip negative LR or infer validity from model name/convergence alone."""
    lr=2*(free.log_likelihood-constrained.log_likelihood)
    df=free.n_parameters-constrained.n_parameters
    reasons=[]
    if not (constrained.converged and free.converged):reasons.append('NOT CONVERGED')
    if not np.isfinite(lr) or lr<0 or df<=0:reasons.append('Geçersiz LR/df')
    for fit in (constrained,free):
        for blocks in fit.standard_errors.values():
            if any(not np.isfinite(v).all() or (np.asarray(v)<0).any() for v in blocks.values()):
                reasons.append('Geçersiz SE')
    if model in ('3PL','4PL'):reasons.append('3PL/4PL sınır/çıkarım kapısı')
    reasons.append('Hessian/SE düzenliliği bağımsız doğrulanmadı; p/q kapalı')
    return float(lr),int(df),'; '.join(dict.fromkeys(reasons))


def _fit_statistics(fit,name):
    return dict(Fit=name,LogLik=fit.log_likelihood,AIC=fit.aic,BIC=fit.bic,
                Parameters=fit.n_parameters,Converged=fit.converged,Iterations=fit.n_iterations)


def check_fit(fit,anchors):
    """Verify observable constraints; no claim that these replace a Hessian."""
    ref=fit.group_labels.index('reference')
    dist=fit.latent_distributions[ref]
    if not np.array_equal(dist.mean,np.zeros(1)) or not np.array_equal(dist.cov,np.ones((1,1))):
        raise ValueError('Referans N(0,1) tanımlaması bozuldu')
    reference=fit.model.get_group_model(ref)
    for gi in range(fit.n_groups):
        engine=fit.model.get_group_model(gi)
        if not all(np.isfinite(v).all() for v in engine.parameters.values()):raise ValueError('Sonlu olmayan parametre')
        if not np.isfinite(fit.latent_distributions[gi].cov).all() or fit.latent_distributions[gi].cov[0,0]<=0:
            raise ValueError('Geçersiz gizil varyans')
        for item in anchors:
            j=fit.model.item_names.index(item)
            a=reference.get_item_parameters(j);b=engine.get_item_parameters(j)
            if any(not np.array_equal(a[name],b[name]) for name in a):raise ValueError('Çapa eşitliği sağlanmadı')


def analyze(data,*,items,group,reference,focal,model,anchors,studied,parameters=('intercepts',),
            correction='fdr_bh',alpha=.05,labels=None,max_iter=500,tol=1e-4,n_quadpts=61):
    import mirt
    from mirt.multigroup.invariance import InvarianceSpec,invariance_lrt
    items=list(items);anchors=list(anchors);studied=list(studied);parameters=tuple(parameters)
    settings=normalize_solver_settings(max_iter=max_iter,tol=tol,n_quadpts=n_quadpts)
    if correction not in ('none','holm','bonferroni','fdr_bh') or not 0<alpha<1:
        raise ValueError('Geçersiz düzeltme/alpha')
    if not parameters or len(set(parameters))!=len(parameters) or not set(parameters)<=set(('intercepts','discrimination')):
        raise ValueError('Yalnız doğrulanmış konum/eğim aileleri')
    if model in ('Rasch','1PL','PCM') and 'discrimination' in parameters:
        raise ValueError('Sabit a=1 modelinde eğim DIF testi yok')
    if len(set(anchors))<2 or len(set(anchors))!=len(anchors) or not set(anchors)<=set(items):
        raise ValueError('En az iki benzersiz, önceden belirlenmiş çapa seçin (yeterlilik garantisi değildir)')
    if not studied or len(set(studied))!=len(studied) or not set(studied)<=set(items)-set(anchors):
        raise ValueError('İncelenen maddeler çapalardan ayrı, benzersiz ve seçili olmalı')
    x,g,lab,freq,excluded=prepare(data,items,group,reference,focal,model,labels)
    mapped='1PL' if model=='Rasch' else model
    def fit(freed):
        ix=[items.index(i) for i in freed] or None
        spec=InvarianceSpec('strict',free_discrimination=ix if 'discrimination' in parameters else None,
                           free_intercepts=ix if 'intercepts' in parameters else None)
        return mirt.fit_multigroup(x,g,model=mapped,invariance=spec,reference_group='reference',
                    n_categories=None if len(lab)==2 else len(lab),**settings)
    freed=[i for i in items if i not in anchors]
    free=fit(freed);check_fit(free,anchors)
    fits={'H1':free};rows=[];diagnostics=[_fit_statistics(free,'H1')]
    for item in studied:
        constrained=fit([i for i in freed if i!=item]);check_fit(constrained,anchors);fits['H0:'+item]=constrained
        lr,df,reason=gate(constrained,free,model)
        # Use the library LR API only when the actual signed statistic is valid.
        if np.isfinite(lr) and lr>=0 and df>0 and constrained.converged and free.converged:
            reference_lr=invariance_lrt(constrained,free)
            if reference_lr['chi2']!=lr or reference_lr['df']!=df:raise ValueError('Kütüphane LR tutarsızlığı')
        rows.append(dict(Item=item,LR=lr,df=df,p=np.nan,q=np.nan,Decision='ÇIKARIM KAPALI',
                         Delta_AIC=constrained.aic-free.aic,Delta_BIC=constrained.bic-free.bic,Reason=reason))
        diagnostics.append(_fit_statistics(constrained,'H0:'+item))
    tables={'DIF':pd.DataFrame(rows),'Fit':pd.DataFrame(diagnostics),'Categories':freq,
            'Latent':pd.DataFrame(free.latent_pars()),'Support':support_matrix()}
    counts=pd.Series(g).value_counts()
    metadata=dict(Model=model,Engine='Python mirt '+mirt.__version__,Python=platform.python_version(),
        Status='NOT CERTIFIED',Design='Explicit-anchor add; selected families free on all nonanchors',
        Reference=repr(reference),Focal=repr(focal),N_reference=int(counts['reference']),N_focal=int(counts['focal']),
        Excluded_other_groups=excluded,Items=items,Anchors=anchors,Studied=studied,Category_labels=lab,
        Parameters=parameters,Requested_correction=correction,Correction_applied=False,Alpha=alpha,Inference=False,**settings,
        Data_SHA256=fingerprint(data[items+[group]],()),
        fit_multigroup_signature=str(inspect.signature(mirt.fit_multigroup)),
        multigroup_dif_signature=str(inspect.signature(mirt.multigroup_dif)),Limitations=LIMITATION)
    tables['Summary']=pd.DataFrame([{'Setting':k,'Value':str(v)} for k,v in metadata.items()])
    coefs=[]
    for gi,name in enumerate(free.group_labels):
        for j,item in enumerate(items):
            for param,value in free.model.get_group_model(gi).get_item_parameters(j).items():
                for coordinate,val in np.ndenumerate(np.asarray(value,float)):
                    coefs.append(dict(Group=name,Item=item,Parameter=param,Coordinate=str(coordinate),Value=float(val)))
    coefficients=pd.DataFrame(coefs)
    tables['All_fit_parameters']=pd.concat([pd.DataFrame(f.coef(include_standard_errors=True)).assign(Fit=name) for name,f in fits.items()],ignore_index=True)
    tables['All_fit_latent']=pd.concat([pd.DataFrame(f.latent_pars()).assign(Fit=name) for name,f in fits.items()],ignore_index=True)
    tables['Parameters']=coefficients
    contrasts=coefficients.pivot(index=['Item','Parameter','Coordinate'],columns='Group',values='Value').reset_index()
    contrasts['Focal_minus_reference']=contrasts.focal-contrasts.reference
    tables['Contrasts']=contrasts
    tables.update(curve_tables(free,items,lab,model))
    return dict(metadata=metadata,tables=tables,fits=fits)


def curve_tables(fit,items,labels,model):
    theta=np.linspace(-4,4,81);curves=[];info=[]
    for gi,name in enumerate(fit.group_labels):
        engine=fit.model.get_group_model(gi)
        p=np.asarray(engine.probability(theta[:,None]),float)
        if p.ndim==2:p=np.stack([1-p,p],axis=-1)
        if p.shape!=(len(theta),len(items),len(labels)) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(-1),1,rtol=0,atol=1e-10):
            raise ValueError('Geçersiz kütüphane kategori olasılıkları')
        for j,item in enumerate(items):
            I=np.asarray(engine.information(theta[:,None],item_idx=j),float).reshape(-1)
            if I.shape!=theta.shape or not np.isfinite(I).all() or (I<0).any():raise ValueError('Geçersiz kütüphane bilgisi')
            info.extend(dict(Group=name,Item=item,Theta=t,Information=v) for t,v in zip(theta,I))
            for c,label in enumerate(labels):
                curves.extend(dict(Group=name,Item=item,Category=label,Theta=t,Probability=v) for t,v in zip(theta,p[:,j,c]))
    curves=pd.DataFrame(curves);information=pd.DataFrame(info)
    difference=curves.pivot(index=['Item','Category','Theta'],columns='Group',values='Probability').reset_index()
    difference['Focal_minus_reference']=difference.focal-difference.reference
    effect=difference.groupby('Item').Focal_minus_reference.agg(lambda v:float(np.abs(v).max())).rename('Max_abs_probability_difference_on_grid').reset_index()
    tables={'Curves':curves,'Difference':difference,'IIF':information,'TIF':information.groupby(['Group','Theta'],as_index=False).Information.sum(),'Effect':effect}
    if model!='NRM':
        scores={str(label):c for c,label in enumerate(labels)}
        scored=curves.assign(Expected=curves.Probability*curves.Category.astype(str).map(scores))
        expected=scored.groupby(['Group','Theta'],as_index=False).Expected.sum()
        dtf=expected.pivot(index='Theta',columns='Group',values='Expected').reset_index()
        dtf['Focal_minus_reference']=dtf.focal-dtf.reference
        tables['DTF']=dtf
    return tables
