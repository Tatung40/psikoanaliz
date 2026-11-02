"""Independent raw-event/count/interval audit, never fits a model."""
import itertools
import json
import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom
from validation.grm_lr_full_mc import experiment as E


def close(a,b):
    if a is None or b is None:assert a is b;return
    np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-12)


def cp(k,n,alpha):
    if not n:return [0.,1.]
    a=alpha/2
    lo=brentq(lambda p:binom.sf(k-1,n,p)-a,0.,1.,xtol=1e-14) if k else 0.
    hi=brentq(lambda p:binom.cdf(k,n,p)-a,0.,1.,xtol=1e-14) if k<n else 1.
    return [lo,hi]


def pair_ci(b,c,n,alpha=.05):
    if not n:return [-1.,1.]
    pi=cp(b+c,n,alpha/2);q=cp(b,b+c,alpha/2)
    corners=[p*(2*x-1) for p in pi for x in q]
    return [min(corners),max(corners)]


def holm_bh(p):
    p=np.asarray(p,float);assert len(p)==2 and np.isfinite(p).all()
    ix=np.argsort(p);v=p[ix];h=np.minimum(1.,np.maximum.accumulate([2*v[0],v[1]]));b=np.minimum(1.,[min(2*v[0],v[1]),v[1]])
    hh=np.empty(2);bb=np.empty(2);hh[ix]=h;bb[ix]=b
    return hh,bb


def binary(x,y):
    n=len(x);b=sum(a and not z for a,z in zip(x,y));c=sum(z and not a for a,z in zip(x,y));m=b+c
    d=(b-c)/n if n else None
    se=math.sqrt(max(0.,(m-(b-c)**2/n)/(n-1)/n)) if n>1 else None
    p=min(1.,float(2*binom.cdf(min(b,c),m,.5))) if m else 1. if n else None
    h=math.sqrt(2*math.log(2/(.05/9))/n) if n else None
    return dict(n=n,b=b,c=c,delta=d,mcse=se,mcnemar_p=p,exact_ci=pair_ci(b,c,n),joint_nine_ci=[max(-1.,d-h),min(1.,d+h)] if n else [-1,1])


def audit(output):
    saved=json.loads((output/'summary.json').read_text());cached=E.cached_records();valid={};seen=set();items={k:{j:dict(raw=0,holm=0,bh=0) for j in ('P09','P10')} for k in ('minus','zero')};attempted={k:0 for k in ('minus','zero')};timed=0
    for r in itertools.chain(E.pilot_records(),E.stream_new(output)):
        key=(r['seed'],r['impact']);assert key not in seen;seen.add(key);attempted[r['impact']]+=1
        assert r['data_sha256']==cached[r['impact']][r['seed']]['data_sha256']
        result=r['result'];timed+=bool(r.get('timeout',False))
        if not result.get('valid',False):continue
        assert E.F.gates(result)['valid']
        p=[row['p'] for row in result['table']];h,b=holm_bh(p)
        close(h,result['holm']);close(b,result['bh'])
        assert bool(np.any(h<.05))==result['holm_event']
        assert bool(np.any(b<.05))==result['bh_event']
        for j,(name,p0,h0,b0) in enumerate(zip(('P09','P10'),p,h,b)):
            assert (p0<.05)==bool(result['raw_rejections'][j])
            for k,v in (('raw',p0<.05),('holm',h0<.05),('bh',b0<.05)):items[r['impact']][name][k]+=int(v)
        valid[key]=dict(holm=result['holm_event'],bh=result['bh_event'],raw=np.mean(np.asarray(p)<.05))
    report=dict(prior_files_verified=E.protect(),attempted=attempted,valid=len(valid),timeouts=timed,per_item=items,conditions={},binary_contrasts={},interactions={},scientific_certification=False)
    for impact in ('minus','zero'):
        values=[v for (seed,k),v in valid.items() if k==impact];n=len(values);events=sum(v['holm'] for v in values);bh=sum(v['bh'] for v in values)
        c=saved['conditions'][impact];assert c['valid']==n and c['holm_events']==events
        close(c['fwer'],events/n if n else None);close(c['bh_global_null_fdr'],bh/n if n else None)
        close(c['fwer_cp975'],cp(events,n,.025));close(c['joint_nine_cp'],cp(events,n,.05/9))
        close(c['fwer_mcse'],math.sqrt((events/n)*(1-events/n)/(n-1)) if n>1 else None)
        close(c['raw_type1'],sum(v['raw'] for v in values)/n if n else None)
        if n>1:close(c['raw_type1_cluster_mcse'],np.std([v['raw'] for v in values],ddof=1)/math.sqrt(n))
        report['conditions'][impact]=dict(n=n,events=events,bh_events=bh,joint_nine_cp=cp(events,n,.05/9))
    contrast=saved['contrasts']
    for impact in ('minus','zero'):
        for method in ('total','rest'):
            seeds=sorted(seed for seed,k in valid if k==impact and cached[impact][seed]['methods'][method]['valid'])
            name=f'GRM-minus-{method}_{impact}'
            report['binary_contrasts'][name]=binary([valid[(s,impact)]['holm'] for s in seeds],[cached[impact][s]['methods'][method]['holm_event'] for s in seeds])
    seeds=sorted(seed for seed,k in valid if k=='zero' and (seed,'minus') in valid)
    report['binary_contrasts']['GRM_zero-minus-minus']=binary([valid[(s,'zero')]['holm'] for s in seeds],[valid[(s,'minus')]['holm'] for s in seeds])
    for name,r in report['binary_contrasts'].items():
        for key,value in r.items():close(value,contrast[name][key])
    names=list(report['binary_contrasts']);p=[report['binary_contrasts'][n]['mcnemar_p'] for n in names]
    if all(v is not None for v in p):
        order=np.argsort(p);adjusted=np.minimum(1.,np.maximum.accumulate(np.array(p)[order]*np.arange(5,0,-1)))
        for j,v in zip(order,adjusted):close(v,contrast[names[j]]['mcnemar_holm_five_p'])
    for method in ('total','rest'):
        q=np.array([[int(valid[(s,'zero')]['holm']),int(cached['zero'][s]['methods'][method]['holm_event']),int(valid[(s,'minus')]['holm']),int(cached['minus'][s]['methods'][method]['holm_event'])] for s in seeds],float).reshape(-1,4)
        n=len(q);coefficient=np.array([1,-1,-1,1]);mean=float(q.mean(0)@coefficient) if n else None
        se=math.sqrt(max(0.,float(coefficient@np.cov(q,rowvar=False,ddof=1)@coefficient)/n)) if n>1 else None
        name='interaction_GRM-minus-'+method;r=contrast[name]
        close(mean,r['mean']);close(se,r['mcse'])
        half=math.sqrt(8*math.log(2/(.05/9))/n) if n else None
        close([max(-2.,mean-half),min(2.,mean+half)] if n else [-2,2],r['joint_nine_ci'])
        def interval(i,j):
            return pair_ci(int(np.sum((q[:,i]==1)&(q[:,j]==0))),int(np.sum((q[:,i]==0)&(q[:,j]==1))),n,.025)
        zi,mi=interval(0,1),interval(2,3)
        close([max(-2.,zi[0]-mi[1]),min(2.,zi[1]-mi[0])],r['exact_paired_ci'])
        assert sum(r['quadruple_table'].values())==n
        report['interactions'][method]=dict(n=n,mean=mean,mcse_from_four_event_covariance=se)
    for name,meta in json.loads((output/'archive_manifest.json').read_text()).items():assert E.Z.digest((output/name).read_bytes())==meta['sha256']
    for name,value in json.loads((output/'metadata.json').read_text())['sources'].items():assert E.Z.digest((E.ROOT/name).read_bytes())==value
    return report
