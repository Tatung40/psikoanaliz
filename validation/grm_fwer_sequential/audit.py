"""Independent raw-evidence audit; never calls an optimizer."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import gammaincc
from scipy.stats import binom
from validation.grm_fwer_sequential.statistics import TAIL_ALPHA,summary
from validation.grm_fwer_sequential.runner import HERE,ROOT,generate,ordinal_hashes


def audit(output=HERE/'results'):
    meta=json.loads((output/'metadata.json').read_text())
    for family in ('sources','archives'):
        for p,h in meta[family].items():
            assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,(family,p)
    rows=[json.loads(v) for v in gzip.decompress((output/'records.jsonl.gz').read_bytes()).splitlines()]
    hashes,_=ordinal_hashes();raw_events={k:[] for k in ('minus','zero')};failures={k:0 for k in raw_events}
    expected=[(2026110100+i,k) for i in range((len(rows)+1)//2) for k in ('minus','zero')]
    assert [(r['seed'],r['impact']) for r in rows]==expected[:len(rows)]
    for r in rows:
        x,g,draws=generate(r['seed'],r['impact'],hashes[r['seed'],r['impact']])
        assert draws==r['rng_draws'] and r['data_sha256']==hashes[r['seed'],r['impact']]
        if not r['valid']:
            assert r['decisions']['p'] is None
            raw_events[r['impact']].append(1);failures[r['impact']]+=1;continue
        p=[]
        for lr in r['lr']:
            assert lr['df']==4 and lr['lr']['161']>=-2e-6
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
        raw_events[r['impact']].append(int(holm))
    result=summary(rows)
    for impact,c in result['conditions'].items():
        n=len(raw_events[impact]);events=sum(raw_events[impact]);assert c['n']==n and c['operational_events']==events
        lo,hi=c['operational_cp']
        if events:assert abs(binom.sf(events-1,n,lo)-TAIL_ALPHA)<1e-10
        if events<n:assert abs(binom.cdf(events,n,hi)-TAIL_ALPHA)<1e-10
        assert c['failures']==failures[impact]
    for seed in range(2026110100,2026110100+len(rows)//2):
        a,b=rows[2*(seed-2026110100):2*(seed-2026110100)+2]
        assert a['rng_draws']==b['rng_draws']
    return dict(status='INDEPENDENT RAW AUDIT PASS',analyses=len(rows),
        sources_and_archives_unchanged=True,all_response_and_draw_hashes_verified=True,
        independent_Holm_BH_and_CP_verified=True,summary=result,
        records_gzip_sha256=hashlib.sha256((output/'records.jsonl.gz').read_bytes()).hexdigest(),
        limitation='Numerical gates checked against saved diagnostics; no new fits or certification.')

if __name__=='__main__':
    from validation.gh_reoptimization.experiment import save
    save(HERE/'results/audit.json',audit())
