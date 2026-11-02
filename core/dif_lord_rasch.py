"""Separate Rasch calibrations: common fixed slope, independent free latent SDs.

Reuse the existing Rasch optimizer/information and Lord contrast; no new estimator.
Legacy fixed-SD numerical/reference functions remain unchanged.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from core import rasch
from core.dif_lord import lord_from_parameters
from core.irt_dif import _checked_covariance

IDENTIFICATION='RASCH_FREE_SD_V1'
STATUS='CONDITIONAL INFERENCE / DIF-FWER NOT CERTIFIED'
ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'validation/lord_rasch_free_scale/results/report.json'


def reference_valid():
    """Fail closed if actual independent evidence is absent or source/data changed."""
    try:
        evidence=json.loads(REFERENCE.read_text())
        required={'core/rasch.py','core/dif_lord.py','core/dif_lord_rasch.py',
                  'tests/fixtures/irt_b10/data_dif1pl_uniform.csv'}
        sources=evidence['production_source_sha256']
        return (evidence['status']=='NUMERICAL REFERENCE PASS / NOT FWER CERTIFIED'
            and evidence['R_execution']=='EXECUTED' and evidence['R_mirt']=='1.48'
            and set(sources)==required
            and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in sources.items()))
    except (OSError,ValueError,KeyError,TypeError):return False


def rasch_physical_fit(fit):
    """b covariance from the FULL inverse information in (b,log SD)."""
    if not fit['converged'] or fit['at_bound']:raise ValueError('Rasch yakınsama/sınır kapısı')
    if fit['se_reason']:raise ValueError(fit['se_reason'])
    b=np.asarray(fit['b'],float);sd=float(fit['sd']);params=np.r_[b,np.log(sd)]
    if (not np.isfinite(params).all() or np.any(np.abs(b)>=rasch.B_BOUND-1e-6)
            or not rasch.LOG_SD_BOUNDS[0]+1e-6<params[-1]<rasch.LOG_SD_BOUNDS[1]-1e-6):
        raise ValueError('Rasch sınırda/geçersiz parametre')
    x,obs=rasch._as_matrix(fit['data']);x1=np.where(obs,x,0.);x0=np.where(obs,1-x,0.)
    z,lw=rasch._grid(fit['points'])
    info=_checked_covariance(rasch._information(params,x1,x0,z,lw))
    full=_checked_covariance(np.linalg.inv(info));bb=_checked_covariance(full[:-1,:-1])
    audit_z,audit_lw=rasch._grid(101)
    score=rasch._score(params,x1,x0,audit_z,audit_lw)/len(x)
    delta=rasch.marginal_loglik(x,b,sd,101)-fit['loglik']
    diagnostics=dict(sd=sd,converged=True,at_bound=False,points=fit['points'],
        normalized_score=fit['max_score_per_person'],GH101_score=float(np.max(np.abs(score))),
        GH101_loglik_difference=float(delta),full_information_condition=float(np.linalg.cond(info)),
        covariance_coordinates='b then log_SD; b block of full inverse information')
    if abs(delta)>1e-3 or diagnostics['GH101_score']>1e-5:raise ValueError('Rasch bağımsız GH101 kapısı')
    return b,bb,full,diagnostics


def lord_1pl_rasch(data,group,reference,focal,*,anchors,studied,points=61,max_iter=2000):
    from core.dif_classical import _prepare
    items=list(data.columns);anchors=list(anchors);studied=list(studied)
    if (len(set(items))!=len(items) or len(set(anchors))<2 or len(set(anchors))!=len(anchors)
            or not set(anchors)<=set(items) or not studied or len(set(studied))!=len(studied)
            or not set(studied)<=set(items)-set(anchors)):
        raise ValueError('En az iki geçerli çapa ve ayrı test maddeleri gerekli')
    x,g,_=_prepare(data,group,reference,focal,False)
    if min(np.bincount(g,minlength=2))<50:raise ValueError('Her grupta en az50 kişi gerekli')
    fits=[];vectors=[];covs=[];full=[];diagnostics=[]
    try:
        for gg in (0,1):
            fit=rasch.fit_rasch(pd.DataFrame(x[g==gg],columns=items),points=points,max_iter=max_iter)
            fits.append(fit);v,c,all_cov,diag=rasch_physical_fit(fit)
            vectors.append(v);covs.append(c);full.append(all_cov);diagnostics.append(diag)
        result=lord_from_parameters(*vectors,*covs,anchors=anchors,studied=studied,items=items,model='1PL')
        result['table']['Lord_identification']=IDENTIFICATION
        # The public presentation policy still requires actual independent reference.
        good=result['table'].p.notna()
        result['table'].loc[good,'Status']=STATUS
        result['table'].loc[good,'Reason']='a=1, μ_R=μ_F=0, σ_R/σ_F serbest; tam (b,log σ) kovaryansı ve çapa delta yöntemi. Genel DIF-FWER sertifikası yok.'
        result.update(fits=fits,vectors=vectors,group_covariances=covs,full_covariances=full,diagnostics=diagnostics)
        return result
    except (ValueError,TypeError,KeyError,np.linalg.LinAlgError) as exc:
        return dict(table=pd.DataFrame([dict(Item=i,Method='Lord χ²',Statistic=np.nan,df=1,p=np.nan,
            Status='UNESTIMABLE / NOT CERTIFIED',Reason=str(exc)) for i in studied]),
            covariance=pd.DataFrame(),fits=fits,diagnostics=diagnostics)
