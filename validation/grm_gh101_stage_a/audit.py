"""Independent raw reader and statistical oracle; optimizers are never invoked."""
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.special import gammaincc
from scipy.stats import binomtest
from core import grm_dif as G
from validation.gh81_diagnostic import diagnostic as D

ROOT=Path(__file__).resolve().parents[2]

def sha(data):return hashlib.sha256(data).hexdigest()

def adjusted(p, method):
    order=sorted(range(len(p)),key=lambda i:p[i]);m=len(p);out=[0.]*m
    if method=='holm':
        previous=0.
        for rank,i in enumerate(order):previous=max(previous,min(1.,p[i]*(m-rank)));out[i]=previous
    else:
        previous=1.
        for rank in range(m-1,-1,-1):
            i=order[rank];previous=min(previous,p[i]*m/(rank+1));out[i]=previous
    return out

def close(a,b):np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-12)

def audit(output):
    start=time.monotonic();meta=json.loads((output/'metadata.json').read_text());summary=json.loads((output/'summary.json').read_text())
    packed=(output/'records.jsonl.gz').read_bytes();raw=gzip.decompress(packed)
    assert sha(raw)==summary['records_sha256'] and sha(packed)==summary['records_gzip_sha256']
    for name,digest in meta['sources'].items():assert sha((ROOT/name).read_bytes())==digest
    for folder,digest in meta['ordinal_sources'].items():assert sha((ROOT/'validation'/folder/'results/records.jsonl.gz').read_bytes())==digest
    rows=[json.loads(line) for line in raw.splitlines()];keys=[(r['seed'],r['impact']) for r in rows]
    plan=json.loads((output.parent/'protocol.json').read_text());order=[(r['seed'],r['impact']) for r in plan['analysis_order']]
    assert keys==order[:len(keys)] and len(set(keys))==len(keys)
    ordinal={(r['seed'],r['impact']):r for r in json.loads((output/'ordinal.json').read_text())}
    report=dict(status='STAGE A / NOT CERTIFIED',rows=len(rows),cold_starts=0,likelihoods=0,gradient_vectors=0,information_matrices=0,lr_records=0,sha256_verified=True,optimizations=0)
    # Independently reconstruct source draws and data equality, including unattempted fixed cases.
    for pairing in meta['pairing']:
        rng=np.random.default_rng(pairing['seed']);draws=[rng.standard_normal(500),rng.standard_normal(500)]+[rng.random(1000) for _ in range(10)]
        assert [sha(v.tobytes()) for v in draws]==[r['sha256'] for r in pairing['draws']]
        with np.load(output/f'matrices/{pairing["seed"]}_minus.npz') as f:minus,gm=f['X'],f['group']
        with np.load(output/f'matrices/{pairing["seed"]}_zero.npz') as f:zero,gz=f['X'],f['group']
        assert np.array_equal(gm,gz) and np.array_equal(minus[:500],zero[:500]) and (zero[500:]>=minus[500:]).all()
        for impact,x in [('minus',minus),('zero',zero)]:assert sha(x.tobytes()+gm.tobytes())==ordinal[pairing['seed'],impact]['data_sha256']
    for r in rows:
        assert sha((output/r['matrix_file']).read_bytes())==r['matrix_sha256']
        with np.load(output/r['matrix_file']) as f:x,g=f['X'],f['group']
        assert sha(x.tobytes()+g.tobytes())==r['data_sha256']==ordinal[r['seed'],r['impact']]['data_sha256']
        calls=r.get('optimizer_calls',[]);assert len({c['solution'] for c in calls})==len(calls)
        for call in calls:
            s=r['solutions'].get(call['solution'])
            if s is None:continue # interrupted hypothesis: retain evidence, no invented completed result
            d=G.Design(s['roles'],4);counts=np.array([(x==i).sum(0) for i in range(4)])
            tails=np.cumsum(counts[:0:-1],axis=0)[::-1].T/len(x);tails=np.minimum(.98,np.maximum(.02,tails));b=-np.log(tails/(1-tails))
            values=[1. if kind=='a' else b[item,0] if c==0 else np.log(b[item,c]-b[item,c-1]) for kind,group,item,c in d.labels]
            lo,hi=np.array(d.bounds).T;expected=np.minimum(hi,np.maximum(lo,np.r_[values,0.,0.]))
            np.testing.assert_array_equal(expected,call['x0']);assert sha(expected.tobytes())==call['expected_cold_sha256']
            assert call['cold_verified'] and call['method']=='L-BFGS-B' and call['jac'] is True
            assert call['options']==dict(maxiter=2000,ftol=1e-14,gtol=1e-10);report['cold_starts']+=1
        for key,s in r['solutions'].items():
            d=G.Design(s['roles'],4);p=np.array(s['params']);a,b,mu,sd=d.unpack(p)
            np.testing.assert_array_equal(a[0,:8],a[1,:8]);np.testing.assert_array_equal(b[0,:8],b[1,:8]);assert (np.diff(b,axis=2)>0).all()
            if 'evaluations' not in s:continue
            parts=G.pattern_parts(x,g,4)
            for q,e in s['evaluations'].items():
                ll=-len(x)*D.reference_nll(p,d,x,g,int(q));assert abs(ll-e['loglik'])<1e-8
                close(ll,e['independent_loglik']);report['likelihoods']+=1
            value,gradient=G.objective(p,d,parts,D.reference_grid(161));close(gradient,s['evaluations']['161']['gradient']);report['gradient_vectors']+=1
            if s['passed']:
                assert s['native_score']<2e-6 and max(abs(gradient))<=1e-5 and s['quadrature_error']<=1e-3
                assert s['converged'] and not s['at_bound'] and s['covariance_available']
                lo,hi=np.array(d.bounds).T;assert np.min(np.minimum(p-lo,hi-p))>=1e-4
                assert s['native_information_min_eigenvalue']>1e-7*max(s['native_information_min_eigenvalue']*s['native_information_condition'],1.)
                for fd in s['derivative_checks'].values():np.testing.assert_allclose(fd['gradient'],gradient,atol=1e-8,rtol=1e-5)
            matrix=np.array(s['dense_information']['matrix']);assert matrix.shape==(d.size,d.size)
            independent=np.empty_like(matrix)
            for j in range(d.size):
                h=1e-5*max(1.,abs(p[j]));step=np.zeros(d.size);step[j]=h
                independent[:,j]=len(x)*(G.objective(p+step,d,parts,D.reference_grid(161))[1]-G.objective(p-step,d,parts,D.reference_grid(161))[1])/(2*h)
            np.testing.assert_allclose(matrix,(independent+independent.T)/2,atol=1e-8,rtol=1e-9)
            eigen=np.linalg.eigvalsh(matrix);close(eigen,s['dense_information']['eigenvalues'])
            if s['passed']:assert eigen[0]>1e-7*max(eigen[-1],1.)
            report['information_matrices']+=1
        for v in r.get('lr',[]):
            null={'omnibus':'H0','thresholds':'Ha','slope':'Hb'}[v['test']]
            alt=r['solutions'][r['slots'][v['item']+'.H2']];base=r['solutions'][r['slots'][v['item']+'.'+null]]
            assert v['df']==G.Design(alt['roles'],4).size-G.Design(base['roles'],4).size=={'omnibus':4,'thresholds':3,'slope':1}[v['test']]
            for q in ('101','161'):
                direct=2*(alt['evaluations'][q]['independent_loglik']-base['evaluations'][q]['independent_loglik'])
                if v['valid']:assert direct>=-2e-6;np.testing.assert_allclose(max(0,direct),v['lr'][q],atol=2e-8,rtol=1e-9)
            report['lr_records']+=1
        if r['valid']:
            assert len(calls)==len(r['solutions'])==7 and all(s['passed'] for s in r['solutions'].values()) and len(r['lr'])==6
            for kind in ('omnibus','thresholds','slope'):
                family=sorted((v for v in r['lr'] if v['test']==kind),key=lambda v:v['item'])
                p=[float(gammaincc(v['df']/2,v['lr']['161']/2)) for v in family];record=r['decisions']['component_families'][kind]
                close(p,record['p'])
                for label,m in [('holm','holm'),('bh','bh')]:close(adjusted(p,m),record[label])
                assert record['holm_event']==any(v<.05 for v in adjusted(p,'holm'))
                assert record['bh_event']==any(v<.05 for v in adjusted(p,'bh'))
        else:assert r['decisions']['p'] is None and r['decisions']['holm_event'] is None
    statistical_audit(rows,ordinal,summary)
    report.update(valid=sum(r['valid'] for r in rows),invalid=sum(not r['valid'] for r in rows),statistical_audit=True,seconds=time.monotonic()-start)
    return report


def statistical_audit(rows,ordinal,summary):
    assert summary['attempted']==len(rows) and summary['valid']==sum(r['valid'] for r in rows)
    assert summary['invalid']==len(rows)-summary['valid'] and summary['not_run']==200-len(rows)
    by={(r['seed'],r['impact']):r for r in rows}
    def event(seed,impact,method):
        v=by.get((seed,impact),{}).get('decisions',{}) if method=='GRM' else ordinal[seed,impact]['methods'][method]
        return int(v['holm_event']) if v.get('valid',False) else None
    for impact,c in summary['conditions'].items():
        valid=[r['decisions'] for r in rows if r['impact']==impact and r['valid']]
        assert c['valid']==len(valid)
        for field,events in [('holm_fwer',[int(v['holm_event']) for v in valid]),('bh_global_null_fdr',[int(v['bh_event']) for v in valid])]:
            result=c[field];n=len(events);k=sum(events);assert result['n']==n and result['events']==k
            if n:
                close(result['rate'],k/n)
                for name,alpha in [('cp95',.05),('cp975',.025),('joint11_cp',.05/11)]:
                    ci=binomtest(k,n).proportion_ci(confidence_level=1-alpha,method='exact');close(result[name],[ci.low,ci.high])
            if n>1:close(result['mcse'],math.sqrt(sum((v-k/n)**2 for v in events)/(n*(n-1))))
            close(result['unknown_failure_empirical_bounds'],[k/100,(k+100-n)/100])
    selectors={}
    for impact in ('minus','zero'):
        for method in ('total','rest'):selectors[f'GRM-minus-{method}_{impact}']=[(impact,'GRM'),(impact,method)]
    selectors['GRM_zero-minus-minus']=[('zero','GRM'),('minus','GRM')]
    for impact in ('minus','zero'):selectors[f'total-minus-rest_{impact}']=[(impact,'total'),(impact,'rest')]
    for method in ('total','rest'):selectors[f'(GRM-{method})_zero-minus-minus']=[('zero','GRM'),('zero',method),('minus','GRM'),('minus',method)]
    for name,parts in selectors.items():
        tuples=[]
        for seed in range(2026110100,2026110200):
            v=tuple(event(seed,*part) for part in parts)
            if all(x is not None for x in v):tuples.append(v)
        result=summary['contrasts'][name];n=len(tuples);width=2 if len(parts)==2 else 4
        values=[v[0]-v[1] if width==2 else v[0]-v[1]-v[2]+v[3] for v in tuples]
        assert result['n']==n and result['missing_from_fixed100']==100-n
        assert result['table']=={''.join(map(str,q)):tuples.count(q) for q in itertools.product((0,1),repeat=len(parts))}
        if n:
            mean=sum(values)/n;close(result['mean'],mean)
            for field,alpha in [('hoeffding_ci',.05),('joint11_ci',.05/11)]:
                half=width*math.sqrt(math.log(2/alpha)/(2*n));close(result[field],[max(-width/2,mean-half),min(width/2,mean+half)])
        if n>1:close(result['mcse'],math.sqrt(sum((v-sum(values)/n)**2 for v in values)/(n*(n-1))))
        close(result['unknown_failure_empirical_bounds'],[(sum(values)-width/2*(100-n))/100,(sum(values)+width/2*(100-n))/100])
        if width==2:
            b=tuples.count((1,0));c=tuples.count((0,1))
            if b+c:close(result['mcnemar_p'],binomtest(min(b,c),b+c,.5).pvalue)
    binary=[n for n,p in selectors.items() if len(p)==2]
    p=[summary['contrasts'][n]['mcnemar_p'] if summary['contrasts'][n]['n'] else 1 for n in binary]
    for label,kind in [('Holm','holm'),('BH','bh')]:
        for name,value in zip(binary,adjusted(p,kind)):
            if summary['contrasts'][name]['n']:close(summary['contrasts'][name][label+'_seven_p'],value)
    def paired_ci(values, i, j, alpha):
        n=len(values)
        if not n:return [-1.,1.]
        b=sum(v[i] and not v[j] for v in values);c=sum(v[j] and not v[i] for v in values)
        def interval(k,n):
            if not n:return [0.,1.]
            ci=binomtest(k,n).proportion_ci(confidence_level=1-alpha/2,method='exact');return [ci.low,ci.high]
        pi=interval(b+c,n);q=interval(b,b+c);corners=[p*(2*v-1) for p in pi for v in q]
        return [min(corners),max(corners)]
    for name,parts in selectors.items():
        values=[]
        for seed in range(2026110100,2026110200):
            v=tuple(event(seed,*p) for p in parts)
            if all(x is not None for x in v):values.append(v)
        if len(parts)==2:expected=paired_ci(values,0,1,.05)
        else:
            z,m=paired_ci(values,0,1,.025),paired_ci(values,2,3,.025)
            expected=[max(-2.,z[0]-m[1]),min(2.,z[1]-m[0])]
        close(summary['contrasts'][name]['exact_paired95'],expected)
    for impact,c in summary['conditions'].items():
        raw=[sum(r['decisions']['raw_rejections'])/2 for r in rows if r['impact']==impact and r['valid']]
        assert c['raw_type1']['n']==len(raw)
        if raw:close(c['raw_type1']['mean'],sum(raw)/len(raw))
        if len(raw)>1:close(c['raw_type1']['mcse'],np.std(raw,ddof=1)/math.sqrt(len(raw)))
        for method,result in c['ordinal'].items():
            values=[event(s,impact,method) for s in range(2026110100,2026110200) if event(s,impact,method) is not None]
            assert result['n']==len(values) and result['events']==sum(values)
