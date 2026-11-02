"""Independent cached-evidence audit; never calls an optimizer."""
import gzip
import json
import time
import numpy as np
from core import grm_dif as G
from validation.grm_gh101_uniform_pilot import pilot as P


def independent_cold_start(d,x):
    counts=np.array([(x==c).sum(axis=0) for c in range(4)])
    tails=np.cumsum(counts[:0:-1],axis=0)[::-1].T/len(x)
    clipped=np.minimum(.98,np.maximum(.02,tails));b=-np.log(clipped/(1-clipped))
    values=[]
    for kind,group,item,c in d.labels:
        values.append(1. if kind=='a' else b[item,0] if c==0 else np.log(b[item,c]-b[item,c-1]))
    lo,hi=np.array(d.bounds).T
    return np.minimum(hi,np.maximum(lo,np.r_[values,0.,0.]))


def audit(output):
    start=time.monotonic();raw=gzip.decompress((output/'records.jsonl.gz').read_bytes())
    rows=[json.loads(line) for line in raw.splitlines()];legacy=P.legacy_records();summary=json.loads((output/'summary.json').read_text());meta=json.loads((output/'metadata.json').read_text())
    assert P.D.sha(raw)==summary['records_sha256']
    assert P.D.sha((output/'records.jsonl.gz').read_bytes())==summary['records_gzip_sha256']
    assert len({(r['seed'],r['impact']) for r in rows})==len(rows)
    for path,digest in meta['sources'].items():assert P.D.sha((P.ROOT/path).read_bytes())==digest
    report=dict(cold_optimizer_calls_verified=0,native_completed_fits=0,valid_datasets_verified=0,LR_grid_records_verified=0,
                independent_likelihood_rechecks=0,maximum_loglik_discrepancy=0.,prior_files_verified=P.protect(),scientific_certification=False,data_generated=0)
    designs,slots=P.layouts()
    for r in rows:
        old=legacy[(r['seed'],r['impact'])];x,g=P.load_matrix(old)
        assert r['data_sha256']==old['data_sha256'];calls=r.get('optimizer_calls',[])
        assert len({c['solution'] for c in calls})==len(calls)
        for call in calls:
            d=designs[call['solution']]['design'];expected=independent_cold_start(d,x)
            np.testing.assert_array_equal(call['x0'],expected)
            assert call['cold_verified'] and call['expected_cold_sha256']==P.D.sha(expected.tobytes())
            assert call['method']=='L-BFGS-B' and call['jac'] is True
            assert call['options']==dict(maxiter=2000,ftol=1e-14,gtol=1e-10)
            assert expected[-2:].tolist()==[0.,0.]
            report['cold_optimizer_calls_verified']+=1
        for key,s in r.get('solutions',{}).items():
            d=designs[key]['design'];p=np.asarray(s['params']);a,b,mu,sd=d.unpack(p)
            np.testing.assert_array_equal(a[0,:8],a[1,:8]);np.testing.assert_array_equal(b[0,:8],b[1,:8]);assert (np.diff(b,axis=2)>0).all()
            if 'evaluations' not in s:continue
            for q,e in s['evaluations'].items():
                independent=-len(x)*P.D.reference_nll(p,d,x,g,int(q));delta=abs(independent-e['loglik'])
                assert delta<1e-8
                report['maximum_loglik_discrepancy']=max(report['maximum_loglik_discrepancy'],delta);report['independent_likelihood_rechecks']+=1
                np.testing.assert_allclose(independent,e['independent_loglik'],atol=1e-8,rtol=0)
                np.testing.assert_allclose(np.max(np.abs(e['gradient'])),e['max_abs_normalized_gradient'],atol=1e-15,rtol=0)
            analytic=np.array(s['evaluations']['161']['gradient'])
            for h,fd in s['derivative_checks'].items():
                error=np.max(np.abs(np.array(fd['gradient'])-analytic));assert error==fd['max_abs_error']
                if s['passed']:np.testing.assert_allclose(fd['gradient'],analytic,atol=1e-8,rtol=1e-5)
            if s['passed']:
                assert s['converged'] and not s['at_bound'] and s['covariance_available'] and s['quadrature_ok']
                assert s['native_score']<2e-6 and s['audit_score']<=1e-5 and s['quadrature_error']<=1e-3
                lo,hi=np.array(d.bounds).T;assert np.min(np.minimum(p-lo,hi-p))>=1e-4
                eigen=np.array(s['dense_information']['eigenvalues']);assert eigen[0]>1e-7*max(eigen[-1],1.)
            item,h=s['first_slot'].split('.');old_fit=old['result']['fits'][item][h];od,op,_=P.D.recover(item,h,old_fit)
            old_ll=-len(x)*P.D.reference_nll(op,od,x,g,161)
            np.testing.assert_allclose(old_ll,s['legacy_comparison']['old_fixed_GH161_loglik'],atol=1e-8,rtol=0)
        for lr in r.get('lr',[]):
            alt=r['solutions'][slots[lr['item']+'.H2']];null={'omnibus':'H0','thresholds':'Ha','slope':'Hb'}[lr['test']];base=r['solutions'][slots[lr['item']+'.'+null]]
            assert lr['df']==designs[slots[lr['item']+'.H2']]['design'].size-designs[slots[lr['item']+'.'+null]]['design'].size
            for q,stat in lr['lr'].items():
                manual=2*(alt['evaluations'][q]['loglik']-base['evaluations'][q]['loglik'])
                if lr['valid']:np.testing.assert_allclose(manual,stat,atol=1e-9,rtol=1e-10)
                report['LR_grid_records_verified']+=1
            old_lr161=2*(alt['legacy_comparison']['old_fixed_GH161_loglik']-base['legacy_comparison']['old_fixed_GH161_loglik'])
            if not lr['legacy_common_GH161_note']:np.testing.assert_allclose(max(0.,old_lr161),lr['legacy_common_GH161_lr'],atol=1e-8,rtol=0)
        report['native_completed_fits']+=r.get('fits_completed',0)
        if r.get('valid',False):
            assert len(calls)==len(r['solutions'])==7 and all(s['passed'] for s in r['solutions'].values()) and all(l['valid'] for l in r['lr'])
            report['valid_datasets_verified']+=1
    assert report['valid_datasets_verified']==summary['valid']
    assert report['native_completed_fits']==summary['unique_hypothesis_fits_completed']
    report['technical_decision_reproduced']=P.decision(rows,summary['stop'])
    assert report['technical_decision_reproduced']==summary['technical_decision']
    report['seconds']=time.monotonic()-start
    return report
