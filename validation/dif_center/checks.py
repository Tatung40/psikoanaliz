"""Deterministic no-fit analytical checks; no RNG, calibration or R execution."""
import inspect
import json
import platform
from pathlib import Path
import mirt
import numpy as np
import pandas as pd
import scipy
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import chi2
from core.dif_center import observed,raju_area
from mirt.multigroup.results import MultigroupFitResult


def run():
    evidence=dict(status='NUMERICAL CHECKS ONLY / NOT CERTIFIED',R_execution='NOT EXECUTED',
        Python=platform.python_version(),mirt=mirt.__version__,numpy=np.__version__,scipy=scipy.__version__,
        sibtest_signature=str(inspect.signature(mirt.sibtest)),wald_signature=str(inspect.signature(mirt.wald)),
        multigroup_fields=list(MultigroupFitResult.__dataclass_fields__),simulations=0,calibrations=0)
    rows=[]
    for group,pattern in [('R',[0,1,1]),('F',[0,0,1])]:
        for anchor in (0,1):rows.extend(dict(A=anchor,B=anchor,Y=y,G=group) for y in pattern*10)
    args=dict(items=['A','B','Y'],group='G',reference='R',focal='F',anchors=['A','B'],studied=['Y'],
              methods=['SIBTEST'],sib_correction=False)
    actual=observed(pd.DataFrame(rows),**args)['table'].iloc[0]
    expected_se=np.sqrt(.5*(2*((2/9)*30/29)/30))
    evidence['SIBTEST_original_uncorrected']=dict(beta=float(actual.beta),se=float(actual.beta_se),
        chi2=float(actual.Statistic),df=int(actual.df),p=float(actual.p),
        max_abs_error=float(max(abs(actual.beta-1/3),abs(actual.beta_se-expected_se),abs(actual.Statistic-14.5),abs(actual.p-chi2.sf(14.5,1)))))
    frame=pd.DataFrame(rows);frame.loc[(frame.G=='R') & (frame.A==1),'Y']=([0,0,1]*10)
    frame.loc[(frame.G=='F') & (frame.A==1),'Y']=([0,1,1]*10)
    crossing=observed(frame,**args,sib_method='crossing')['table'].iloc[0]
    evidence['SIBTEST_crossing_uncorrected']=dict(beta_cross=float(crossing.beta),chi2=float(crossing.Statistic),
        df=int(crossing.df),p=float(crossing.p),expected_chi2=14.5,expected_p=float(chi2.sf(14.5,2)))
    null=pd.DataFrame(rows);null.loc[null.G=='F','Y']=null.loc[null.G=='R','Y'].to_numpy()
    null_result=observed(null,**args)['table'].iloc[0]
    evidence['SIBTEST_exact_balanced_null']=dict(beta=float(null_result.beta),p=float(null_result.p),
        note='One deterministic identity, NOT a Type-I/FWER experiment')
    areas=[]
    for r,f in [([1,-.4,0,1],[1,.5,0,1]),([.8,-.5,0,1],[1.6,.7,0,1]),([1.2,.3,.15,.9],[.7,-.4,.15,.9])]:
        value=lambda t:(f[2]+(f[3]-f[2])*expit(f[0]*(t-f[1])))-(r[2]+(r[3]-r[2])*expit(r[0]*(t-r[1])))
        signed=quad(value,-np.inf,np.inf,epsabs=1e-11)[0]
        cross=(f[0]*f[1]-r[0]*r[1])/(f[0]-r[0]) if f[0]!=r[0] else 0.
        unsigned=sum(quad(lambda t:abs(value(t)),a,b,epsabs=1e-11)[0] for a,b in [(-np.inf,cross),(cross,np.inf)])
        actual=raju_area(r,f)
        areas.append(dict(reference=r,focal=f,signed=actual['Signed_area'],unsigned=actual['Unsigned_area'],
                          signed_error=abs(actual['Signed_area']-signed),unsigned_error=abs(actual['Unsigned_area']-unsigned)))
    evidence['Raju_adaptive_quadrature']=areas
    assert evidence['SIBTEST_original_uncorrected']['max_abs_error']<1e-10
    assert abs(crossing.Statistic-14.5)<1e-10 and crossing.df==2 and null_result.p==1
    assert max(max(r['signed_error'],r['unsigned_error']) for r in areas)<1e-10
    return evidence


if __name__=='__main__':
    target=Path(__file__).with_name('checks.json')
    target.write_text(json.dumps(run(),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(str(target))
