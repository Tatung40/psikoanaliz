"""Declared common-a 1PL covariance coordinates; no estimation or pseudo-inverse."""
from types import SimpleNamespace
import numpy as np


def evidence_scope(fit):
    """Fail closed: the committed campaign supports only its declared scope."""
    import hashlib,json,importlib.metadata
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    try:
        evidence=json.loads((root/'validation/evidence/p9_1pl_common_se/coverage.json').read_text())
        options=fit.refit_recipe.options
        return (evidence['status']=='CERTIFIED' and fit.converged
                and fit.n_observations==1000 and len(fit.coef())==12
                and all(options[k]==v for k,v in {'n_quadpts':21,'tol':1e-4,'max_iter':500}.items())
                and options.get('item_priors') is None
                and fit.refit_recipe.latent_density is None
                and (fit.latent_mean is None or np.array_equal(fit.latent_mean,np.zeros(1)))
                and (fit.latent_covariance is None or np.array_equal(fit.latent_covariance,np.eye(1)))
                and importlib.metadata.version('mirt')==evidence['versions']['mirt']
                and all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h
                        for p,h in evidence['source_sha256'].items()))
    except (OSError,ValueError,TypeError,KeyError,AttributeError):return False


def common_a_basis(fit):
    options=getattr(getattr(fit,'refit_recipe',None),'options',{})
    groups=options.get('constraints',()) if isinstance(options,dict) else ()
    if not groups:return None
    coef=fit.coef();k=len(coef)
    if (len(groups)!=1 or getattr(groups[0],'parameter',None)!='discrimination'
            or tuple(getattr(groups[0],'items',()))!=tuple(range(k))
            or getattr(groups[0],'column',None) is not None or fit.n_parameters!=k+1):
        raise ValueError('Ortak-a kısıtları/serbest parametre sayısı doğrulanamadı.')
    covariance=np.asarray(fit.vcov() if callable(fit.vcov) else fit.vcov,float)
    labels=list(fit.vcov_labels() if callable(fit.vcov_labels) else fit.vcov_labels)
    if covariance.shape!=(len(labels),len(labels)):raise ValueError('Kovaryans boyutu geçersiz.')
    a=[i for i,s in enumerate(labels) if s.startswith('discrimination[')]
    b=[i for i,s in enumerate(labels) if s.startswith('difficulty[')]
    if len(a)!=k or len(b)!=k or len(labels)!=2*k:raise ValueError('Ortak-a koordinatları geçersiz.')
    if not np.array_equal(coef.discrimination.to_numpy(),np.full(k,coef.discrimination.iloc[0])):
        raise ValueError('Beyan edilen ortak-a eşitliği sağlanmadı.')
    for j in a[1:]:
        if (not np.array_equal(covariance[j],covariance[a[0]],equal_nan=True) or
                not np.array_equal(covariance[:,j],covariance[:,a[0]],equal_nan=True)):
            raise ValueError('Beyan edilen ortak-a kovaryans kopyaları eşit değil.')
    from mirt.estimation.base import _ITEM_PARAMETER_BOUNDS
    for name in ('discrimination','difficulty'):
        lo,hi=_ITEM_PARAMETER_BOUNDS[name]
        v=coef[name].to_numpy(float)
        if not np.isfinite(v).all() or np.any(np.minimum(abs(v-lo),abs(v-hi))<=1e-6):
            raise ValueError('Ortak-a modelinde sınır/geçersiz parametre; Wald çıkarımı kapalı.')
    indices=[a[0],*b]
    return covariance[np.ix_(indices,indices)],[labels[i] for i in indices]


def parameter_inference(fit,model,item_names,estimation=None):
    """Delegate every gate to existing inference on the true independent basis.

    Shared a is returned once (first item's label); its cross-covariance with ALL
    difficulties is retained. Equal numerical estimates alone never authorize reduction.
    """
    from core.irt_binary_inference import parameter_inference as original
    if model!='1PL':return original(fit,model,item_names,estimation)
    try:reduced=common_a_basis(fit)
    except (ValueError,TypeError,AttributeError) as exc:return None,str(exc)
    if reduced is None:return original(fit,model,item_names,estimation)
    if estimation is not None and (estimation!='ML' or not evidence_scope(fit)):
        return None,'Ortak-a Q4 SE/GA kanıtı bu kapsam için sertifikalı değil; çıkarım kapalı.'
    covariance,labels=reduced
    view=SimpleNamespace(converged=fit.converged,vcov=covariance,vcov_labels=labels,coef=fit.coef)
    table,reason=original(view,model,item_names,estimation)
    if table is not None:
        table.loc[table.Parametre=='a','Madde']='Ortak (tüm maddeler)'
        table.loc[table.Parametre=='a','Not']='Tek ortak serbest a; madde sayısı kadar bağımsız aralık değildir.'
    return table,reason
