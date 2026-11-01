"""Opt-in impact0 control: reuse PR120 runner/estimators; never refit impact−.5."""
import gzip
import hashlib
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.special import expit
from scipy.stats import binom
from core.dif_classical import adjust_pvalues
from validation.dif_validation_study import gen_ordinal
from validation.ordinal_dif_paired.control import TraceRNG
from validation.ordinal_dif_paired.experiment import binomial_ci, clean
from validation.ordinal_dif_total_rest import experiment as E

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=HERE.parent/'ordinal_dif_total_rest'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def protect():
    manifest=json.loads((HERE/'pr120_manifest.json').read_text())
    for name,expected in manifest.items():
        if digest((ROOT/name).read_bytes())!=expected:
            raise ValueError(f'Protected PR120 file changed: {name}')
    return len(manifest)


def data_hash(x,g):
    return digest(x.to_numpy(dtype=np.uint8).tobytes()+np.asarray(g,dtype=np.uint8).tobytes())


def paired_control(seed, old_row, nr=500, nf=500):
    trace=TraceRNG(seed)
    with patch('numpy.random.default_rng',return_value=trace):
        x,g=gen_ordinal(nr,nf,0.,False,seed)
    expected=[('standard_normal',[nr]),('standard_normal',[nf])]+[('random',[nr+nf])]*10
    if [(v['method'],v['shape']) for v in trace.calls]!=expected:
        raise ValueError('RNG consumption mismatch: common seed alone is insufficient')
    theta=np.r_[trace.values[0],trace.values[1]-.5]
    negative=np.zeros_like(x.to_numpy())
    for j,a in enumerate(np.linspace(.9,1.8,10)):
        thresholds=np.linspace(-1,1,10)[j]+np.array([-1.,0.,1.])
        surv=expit(a*(theta[:,None]-thresholds[None,:]))
        negative[:,j]=(trace.values[j+2][:,None]<surv).sum(axis=1)
    reconstructed=digest(negative.astype(np.uint8).tobytes()+g.astype(np.uint8).tobytes())
    if reconstructed!=old_row['data_sha256']:
        raise ValueError('Common draws do not reconstruct archived negative-impact data hash')
    reference_equal=np.array_equal(x.to_numpy()[:nr],negative[:nr])
    monotone=bool(np.all(x.to_numpy()[nr:]>=negative[nr:]))
    if not reference_equal or not monotone:
        raise ValueError('Reference equality or focal monotonicity mismatch')
    return x,g,dict(seed=seed,draws=trace.calls,control_hash=data_hash(x,g),
        negative_hash=reconstructed,reference_rows_identical=reference_equal,
        reference_hash=digest(negative[:nr].astype(np.uint8).tobytes()),
        negative_focal_normal_shift=-.5,control_focal_normal_shift=0.,
        focal_responses_monotone=monotone,reconstructed_negative_exactly=True)


def exact_paired_ci(b,c,n,alpha=.05):
    if not n:return [-1.,1.]
    pi=binomial_ci(b+c,n,alpha/2)
    q=binomial_ci(b,b+c,alpha/2)
    corners=[p*(2*v-1) for p in pi for v in q]
    return [min(corners),max(corners)]


def mean_stats(values,width,alpha=.05):
    z=np.asarray(values,float);n=len(z)
    if not n:return dict(n=0,mean=None,mcse=None,hoeffding_ci=[-width/2,width/2])
    value=float(z.mean());margin=width*math.sqrt(math.log(2/alpha)/(2*n))
    return dict(n=n,mean=value,mcse=float(z.std(ddof=1)/math.sqrt(n)) if n>1 else None,
                hoeffding_ci=[max(-width/2,value-margin),min(width/2,value+margin)])


def contrast(x,y,attempted):
    r=E.paired_stats(x,y)
    n=r['n'];missing=attempted-n
    diff=r.get('b',0)-r.get('c',0)
    r['unknown_failure_empirical_bounds']=[(diff-missing)/attempted,(diff+missing)/attempted] if attempted else [-1,1]
    r['simultaneous_four_contrasts_ci']=mean_stats(np.asarray(x,int)-np.asarray(y,int),2,.05/4)['hoeffding_ci']
    return r


def compare(zero,minus):
    old={r['seed']:r for r in minus}
    if len(old)!=len(minus) or len({r['seed'] for r in zero})!=len(zero):
        raise ValueError('Duplicate seed')
    for r in zero:
        if r['seed'] not in old:raise ValueError('Missing paired archived seed')
    def valid(r,k):return r['methods'].get(k,{}).get('valid',False)
    def event(r,k):return r['methods'][k]['holm_event']
    within=[r for r in zero if all(valid(r,k) for k in ('total','rest'))]
    result=dict(primary_zero_total_minus_rest=contrast([event(r,'total') for r in within],[event(r,'rest') for r in within],len(zero)),impact_changes={})
    for k in ('total','rest'):
        rows=[r for r in zero if valid(r,k) and valid(old[r['seed']],k)]
        result['impact_changes'][k]=contrast([event(r,k) for r in rows],[event(old[r['seed']],k) for r in rows],len(zero))
    complete=[r for r in within if all(valid(old[r['seed']],k) for k in ('total','rest'))]
    quadruples=[tuple(int(event(v,k)) for v in (r,old[r['seed']]) for k in ('total','rest')) for r in complete]
    d0=np.array([a-b for a,b,c,d in quadruples],int)
    dm=np.array([c-d for a,b,c,d in quadruples],int)
    interaction=mean_stats(d0-dm,4)
    interaction['definition']='(total-rest)_impact0 - (total-rest)_impact_minus0.5'
    interaction['simultaneous_four_contrasts_ci']=mean_stats(d0-dm,4,.05/4)['hoeffding_ci']
    n=len(quadruples)
    z=E.paired_stats([a for a,b,c,d in quadruples],[b for a,b,c,d in quadruples])
    m=E.paired_stats([c for a,b,c,d in quadruples],[d for a,b,c,d in quadruples])
    zi=exact_paired_ci(z.get('b',0),z.get('c',0),n,.025)
    mi=exact_paired_ci(m.get('b',0),m.get('c',0),n,.025)
    interaction['conservative_exact_paired_ci']=[max(-2.,zi[0]-mi[1]),min(2.,zi[1]-mi[0])]
    interaction['signed_event_table']={f'{a},{b}':int(np.sum((d0==a)&(dm==b))) for a,b in itertools.product((-1,0,1),repeat=2)}
    interaction['quadruple_order']=['T0','R0','Tm','Rm']
    interaction['quadruple_table']={''.join(map(str,q)):quadruples.count(q) for q in itertools.product((0,1),repeat=4)}
    missing=len(zero)-n;s=float(np.sum(d0-dm))
    interaction['unknown_failure_empirical_bounds']=[(s-2*missing)/len(zero),(s+2*missing)/len(zero)] if zero else [-2,2]
    interaction['mcnemar']='NOT APPLICABLE: difference of paired differences is not a binary paired event contrast'
    result['interaction']=interaction
    contrasts=[result['primary_zero_total_minus_rest'],*result['impact_changes'].values()]
    if all(r['mcnemar_p'] is not None for r in contrasts):
        for r,p in zip(contrasts,adjust_pvalues([r['mcnemar_p'] for r in contrasts],'Holm')):
            r['mcnemar_holm_three_tests_p']=float(p)
    result['all_four_cells_valid']=n
    result['confirmatory_fixed_design_complete']=len(zero)==2500 and n==2500
    result['multiplicity']='Four simultaneous Hoeffding intervals jointly>=95%; individual exact95% intervals not joint. Holm McNemar3 binary contrasts; no McNemar for interaction. Separate FWER CP family within each condition only.'
    result['no_negative_refits']=True
    return clean(result)


def audit_results(zero, minus, comparison):
    """Independent integer/count and covariance reconstruction, no model fits."""
    old={r['seed']:r for r in minus}; events=[]; checked=0
    for r in zero:
        for method in r['methods'].values():
            if not method.get('valid',False):continue
            p=np.array([z['p_omnibus'] for z in method['raw']],float)
            ordered=np.sort(p)
            holm=np.minimum(1.,np.maximum.accumulate([2*ordered[0],ordered[1]]))
            bh=np.minimum(1.,[min(2*ordered[0],ordered[1]),ordered[1]])
            assert bool(np.any(holm<.05))==method['holm_event']
            assert bool(np.any(bh<.05))==method['bh_event']
            checked+=1
        o=old[r['seed']]
        if all(v['methods'].get(k,{}).get('valid',False) for v in (r,o) for k in ('total','rest')):
            events.append([int(v['methods'][k]['holm_event']) for v in (r,o) for k in ('total','rest')])
    data=np.asarray(events,int).reshape(-1,4);n=len(data)
    audit=dict(raw_zero_method_decisions_checked=checked,complete_quadruples=n,
               protected_pr120_files=protect(),scientific_certification=False,contrasts={})
    definitions=[('zero_total_minus_rest',data[:,0],data[:,1],comparison['primary_zero_total_minus_rest']),
                 ('total_impact_change',data[:,0],data[:,2],comparison['impact_changes']['total']),
                 ('rest_impact_change',data[:,1],data[:,3],comparison['impact_changes']['rest'])]
    if n==len(zero):
        for name,x,y,r in definitions:
            b=int(np.sum((x==1)&(y==0)));c=int(np.sum((x==0)&(y==1)));m=b+c
            delta=(b-c)/n
            variance=((b+c)-(b-c)**2/n)/(n-1) if n>1 else None
            exact_p=min(1.,float(2*binom.cdf(min(b,c),m,.5))) if m else 1.
            assert (b,c)==(r['b'],r['c'])
            assert math.isclose(delta,r['delta'],abs_tol=1e-14)
            if n>1:assert math.isclose(math.sqrt(variance/n),r['mcse'],rel_tol=1e-12,abs_tol=1e-14)
            assert math.isclose(exact_p,r['mcnemar_p'],rel_tol=1e-12,abs_tol=1e-300)
            audit['contrasts'][name]=dict(b=b,c=c,delta=delta,mcse=math.sqrt(variance/n) if n>1 else None,mcnemar_from_binomial_cdf=exact_p)
        coefficient=np.array([1,-1,-1,1])
        value=float(data.mean(axis=0)@coefficient)
        mcse=math.sqrt(float(coefficient@np.cov(data,rowvar=False,ddof=1)@coefficient)/n) if n>1 else None
        assert math.isclose(value,comparison['interaction']['mean'],abs_tol=1e-14)
        if n>1:assert math.isclose(mcse,comparison['interaction']['mcse'],rel_tol=1e-12,abs_tol=1e-14)
        audit['interaction_from_four_cell_covariance']=dict(mean=value,mcse=mcse)
    return audit


def load_records(path):
    return [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]


def execute():
    protected=protect()
    old_records=load_records(OLD/'results/records.jsonl.gz')
    old={r['seed']:r for r in old_records}
    meta=json.loads((OLD/'results/metadata.json').read_text())
    for name,version in meta['versions'].items():
        if importlib.metadata.version(name)!=version:raise ValueError(f'Cross-condition version mismatch: {name}')
    for name in ('validation/dif_validation_study.py','core/dif_classical.py'):
        if digest((ROOT/name).read_bytes())!=meta['sources'][name]:raise ValueError(f'Cross-condition source mismatch: {name}')
    if len(old)!=2500 or sorted(old)!=list(range(2026110100,2026112600)):
        raise ValueError('Archived fixed2500 seed family mismatch')
    output=HERE/'results';output.mkdir(exist_ok=True)
    crn_path=output/'crn.jsonl'
    if crn_path.exists() or (output/'crn.jsonl.gz').exists():raise RuntimeError('Existing CRN evidence protected')
    with crn_path.open('x') as stream:
        def zero_generator(nr,nf,mu,dif,seed):
            # PR120 runner supplies its old literal; this adapter explicitly changes ONLY impact.
            if (nr,nf,mu,dif)!=(500,500,-.5,False):raise ValueError('Unexpected reused runner design')
            x,g,trace=paired_control(seed,old[seed],nr,nf)
            stream.write(json.dumps(trace)+'\n');stream.flush()
            return x,g
        try:
            with patch.multiple(E,HERE=HERE,gen_ordinal=zero_generator):E.execute()
        finally:
            protect()
    raw=crn_path.read_bytes();packed=gzip.compress(raw,mtime=0)
    assert gzip.decompress(packed)==raw
    (output/'crn.jsonl.gz').write_bytes(packed);crn_path.unlink()
    meta_path=output/'metadata.json';new_meta=json.loads(meta_path.read_text())
    for path in (Path(__file__),HERE/'plan.json',HERE/'pr120_manifest.json',ROOT/'validation/ordinal_dif_paired/control.py'):
        new_meta['sources'][str(path.relative_to(ROOT))]=digest(path.read_bytes())
    new_meta.update(actual_impact=0.,adapter='Explicit validation-only runner adapter; production function unchanged',
        crn_sha256=digest(raw),crn_gzip_sha256=digest(packed),protected_pr120_files=protected,
        negative_records_sha256=digest((OLD/'results/records.jsonl.gz').read_bytes()))
    E.save(meta_path,new_meta)
    zero=load_records(output/'records.jsonl.gz')
    comparison=compare(zero,old_records)
    E.save(output/'comparison.json',comparison)
    E.save(output/'audit.json',audit_results(zero,old_records,comparison))
    print(json.dumps(comparison,indent=2),flush=True)


if __name__=='__main__':execute()
