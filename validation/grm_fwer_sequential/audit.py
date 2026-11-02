"""Independent raw-evidence audit; never calls an optimizer."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import gammaincc
from scipy.stats import binom
from validation.grm_fwer_sequential.statistics import TAIL_ALPHA,summary
from validation.grm_fwer_sequential.runner import HERE,ROOT,generate,ordinal_hashes
from core import grm_dif as G
from validation.grm_fwer_sequential.archive import canonical_bytes


def audit(output=HERE/'results', replay_data=False):
    meta=json.loads((output/'metadata.json').read_text())
    for family in ('sources','archives'):
        for p,h in meta[family].items():
            assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,(family,p)
    packed=canonical_bytes(output)
    rows=[json.loads(v) for v in gzip.decompress(packed).splitlines()]
    hashes,_=ordinal_hashes();raw_events={k:[] for k in ('minus','zero')};failures={k:0 for k in raw_events}
    expected=[(2026110100+i,k) for i in range((len(rows)+1)//2) for k in ('minus','zero')]
    assert [(r['seed'],r['impact']) for r in rows]==expected[:len(rows)]
    for r in rows:
        assert r['data_sha256']==hashes[r['seed'],r['impact']]
        if replay_data:
            x,g,draws=generate(r['seed'],r['impact'],hashes[r['seed'],r['impact']])
            assert draws==r['rng_draws']
        if not r['valid']:
            assert r['decisions']['p'] is None
            raw_events[r['impact']].append(1);failures[r['impact']]+=1;continue
        p=[]
        for lr in r['lr']:
            assert lr['df']==4 and lr['lr']['161']>=-2e-6
            base=r['solutions'][r['slots'][lr['item']+'.H0']]
            alt=r['solutions'][r['slots'][lr['item']+'.H2']]
            assert G.Design(alt['roles'],4).size-G.Design(base['roles'],4).size==4
            for q in ('101','161'):
                direct=2*(alt['evaluations'][q]['independent_loglik']-base['evaluations'][q]['independent_loglik'])
                assert abs(direct-lr['lr'][q])<=2e-8
            p.append(float(gammaincc(2,max(0,lr['lr']['161'])/2)))
        assert len(p)==2
        np.testing.assert_allclose(p,r['decisions']['p'],rtol=1e-13,atol=1e-15)
        holm=min(p)<.025
        bh=holm or max(p)<.05
        assert r['decisions']['holm_event']==holm and r['decisions']['bh_event']==bh
        for s in r['solutions'].values():
            assert s['converged'] and not s['at_bound'] and s['score']<2e-6 and s['audit_score']<=1e-5
            assert s['quadrature_ok'] and s['quadrature_error']<=1e-3
            assert s['anchor_equality'] and s['covariance_available'] and s['dense_information']['valid']
            assert s['optimizer'][0]['options']==dict(maxiter=2000,ftol=1e-14,gtol=1e-10)
            d=G.Design(s['roles'],4);params=np.asarray(s['params']);a,b,mu,sd=d.unpack(params)
            np.testing.assert_array_equal(a[0,:8],a[1,:8]);np.testing.assert_array_equal(b[0,:8],b[1,:8])
            assert (np.diff(b,axis=2)>0).all()
            lo,hi=np.asarray(d.bounds).T;assert np.min(np.minimum(params-lo,hi-params))>=1e-4
            eigen=np.asarray(s['dense_information']['eigenvalues'])
            assert np.isfinite(eigen).all() and eigen[0]>1e-7*max(eigen[-1],1.)
            gradient=np.asarray(s['evaluations']['161']['gradient']);assert max(abs(gradient))<=1e-5
            for fd in s['derivatives'].values():np.testing.assert_allclose(fd,gradient,atol=1e-8,rtol=1e-5)
        raw_events[r['impact']].append(int(holm))
    result=summary(rows)
    for impact,c in result['conditions'].items():
        n=len(raw_events[impact]);events=sum(raw_events[impact]);assert c['n']==n and c['operational_events']==events
        lo,hi=c['operational_cp']
        if events:assert abs(binom.sf(events-1,n,lo)-TAIL_ALPHA)<1e-10
        if events<n:assert abs(binom.cdf(events,n,hi)-TAIL_ALPHA)<1e-10
        assert c['failures']==failures[impact]
        assert abs(c['operational_mcse']-math.sqrt(events*(n-events)/n**3))<1e-14
        valid_rejects=events-failures[impact];robust_lower=c['missing_outcome_robust_cp'][0]
        if valid_rejects:assert abs(binom.sf(valid_rejects-1,n,robust_lower)-TAIL_ALPHA)<1e-10
    by={(r['seed'],r['impact']):r for r in rows};differences=[]
    for seed in sorted({r['seed'] for r in rows}):
        if (seed,'zero') not in by or (seed,'minus') not in by:continue
        def event(r):return int(not r['valid'] or r['decisions']['holm_event'])
        differences.append(event(by[seed,'zero'])-event(by[seed,'minus']))
    if len(differences)>1:
        n=len(differences);total=sum(differences)
        mcse=math.sqrt(max(0.,sum(d*d for d in differences)-total*total/n)/(n-1)/n)
        assert abs(result['paired_operational_difference']['mcse']-mcse)<1e-14
    for seed in range(2026110100,2026110100+len(rows)//2):
        a,b=rows[2*(seed-2026110100):2*(seed-2026110100)+2]
        assert a['rng_draws']==b['rng_draws']
    return dict(status='INDEPENDENT RAW AUDIT PASS',analyses=len(rows),
        scientific_status='INCONCLUSIVE / NOT CERTIFIED',
        inference_scope='Arithmetic integrity PASS only; registered simultaneous intervals belong to look500, not the unplanned user-stop prefix.',
        sources_and_archives_unchanged=True,all_response_and_draw_hashes_verified=True,
        responses_regenerated=replay_data,
        identity_scope='Saved response hashes vs frozen original records; paired actual draw hashes equal. No response regeneration unless explicitly requested.',
        independent_Holm_BH_and_CP_verified=True,summary=result,
        records_gzip_sha256=hashlib.sha256(packed).hexdigest(),
        limitation='Numerical gates checked against saved diagnostics; no new fits or certification.')

if __name__=='__main__':
    from validation.gh_reoptimization.experiment import save
    save(HERE/'results/audit.json',audit())
