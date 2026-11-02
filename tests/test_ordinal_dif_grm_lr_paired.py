import copy
import json
import numpy as np
import pytest
from core import grm_dif as G
from validation.ordinal_dif_grm_lr_paired import experiment as X


def test_oracle_anchors_identification_and_df_without_fitting():
    roles=['anchor']*8+['ab']*2
    full=G.Design(roles,4)
    assert full.size==50
    for target in (8,9):
        null=roles.copy();null[target]='anchor';d=G.Design(null,4)
        assert full.size-d.size==4
    assert np.array_equal(full.a[0,:8],full.a[1,:8])
    assert np.array_equal(full.b[0,:8],full.b[1,:8])
    a=np.ones((2,10));b=np.broadcast_to([-1.,0.,1.],(2,10,3)).copy()
    aa,bb,mu,sd=full.unpack(full.encode(a,b,-.5,1.))
    np.testing.assert_array_equal(aa[0,:8],aa[1,:8])
    np.testing.assert_array_equal(bb[0,:8],bb[1,:8])
    assert mu==-.5 and sd==1.


def reference_result():
    f=dict(a=np.ones((2,10)).tolist(),b=np.broadcast_to([-1.,0.,1.],(2,10,3)).tolist(),
           focal_mean=-.5,focal_sd=1.,information_min_eigenvalue=2.,information_condition=100.,
           information_available=True,converged=True,at_bound=False,score=1e-8,
           quadrature_ok=True,quadrature_error=1e-6,audit_score=1e-8)
    return dict(valid=True,fits={item:{h:copy.deepcopy(f) for h in ('H0','Hb','Ha','H2')} for item in ('P09','P10')},
                table=[{'Madde':item,'sd (genel)':4,'logL (H2)':-100.,'logL (H0)':-100.,'LR (genel)':0.,'p':1.} for item in ('P09','P10')])


def test_gate_audit_rejects_bad_anchors_information_quadrature_and_df():
    baseline=reference_result();assert X.gates(baseline)['valid']
    for field,value in [('information_available',False),('quadrature_ok',False),('at_bound',True),('converged',False)]:
        r=copy.deepcopy(baseline);r['fits']['P09']['H0'][field]=value
        assert not X.gates(r)['valid']
    r=copy.deepcopy(baseline);r['fits']['P10']['H2']['a'][1][0]=1.1
    assert not X.gates(r)['valid']
    r=copy.deepcopy(baseline);r['table'][0]['sd (genel)']=3
    assert not X.gates(r)['valid']


def test_plan_and_canonical_prior_protection():
    p=json.loads((X.HERE/'plan.json').read_text())
    assert p['planned_grm_analyses']==20 and p['feasibility_seed_pairs']==10
    assert p['budget_seconds']==600 and p['per_analysis_timeout_seconds']==90
    assert p['anchors']==X.ANCHORS and p['categories']==4
    assert '9-component' in p['multiplicity'] and 'feasibility ONLY' in p['decision']
    assert X.protect()==p['preserved_files']==81


def test_saved_feasibility_records_gates_and_pairing_without_refits():
    import gzip
    out=X.HERE/'results'
    packed=(out/'records.json.gz').read_bytes();raw=gzip.decompress(packed)
    saved=json.loads(raw);records=saved['records']
    summary=json.loads((out/'summary.json').read_text());meta=json.loads((out/'metadata.json').read_text())
    assert X.Z.digest(raw)==summary['records_sha256']
    assert X.Z.digest(packed)==summary['records_gzip_sha256']
    assert X.protect()==81 and not summary['main_executed']
    from validation.archived_source_identity import source_matches
    for name,value in meta['sources'].items():assert source_matches(X.ROOT/name,value),name
    cached={k:{r['seed']:r for r in X.Z.load_records(path/'results/records.jsonl.gz')} for k,path in (('minus',X.Z.OLD),('zero',X.Z.HERE))}
    rebuilt=X.summarize(records,cached,summary['seconds'],summary['stop'])
    for key,value in rebuilt.items():assert summary[key]==value
    for r in records:
        assert 2026110100<=r['seed']<=2026110109
        assert r['data_sha256']==cached[r['impact']][r['seed']]['data_sha256']
        path=out/r['matrix_file']
        assert X.Z.digest(path.read_bytes())==r['matrix_sha256']
        with np.load(path) as matrix:
            assert matrix['X'].shape==(1000,10)
            assert X.Z.digest(matrix['X'].tobytes()+matrix['group'].tobytes())==r['data_sha256']
        if r['result'].get('valid',False):
            assert r['result']['gate_audit']==X.gates(r['result'])
            decision=X.P.classify([v['p'] for v in r['result']['table']],.05)
            for key,value in decision.items():assert r['result'][key]==value
        assert r['crn']['reconstructed_negative_exactly']
        assert r['crn']['negative_hash']==cached['minus'][r['seed']]['data_sha256']
    assert summary['seconds']<=600
