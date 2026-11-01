import numpy as np
import pytest
from core import grm_dif as G
from validation.gh81_diagnostic import diagnostic as D


def fixture():
    d=G.Design(['anchor','anchor','ab'],4)
    a=np.array([[1.,1.2,1.4],[1.,1.2,1.3]])
    b=np.broadcast_to([-1.,0.,1.],(2,3,3)).copy();b[1,2]+=.1
    p=d.encode(a,b,-.5,.9)
    x=np.array([[0,1,3],[1,2,0],[3,1,2],[2,0,1]])
    return d,p,x,np.array([0,0,1,1])


def test_independent_likelihood_and_finite_difference_no_fit(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('No fitting allowed')
    monkeypatch.setattr(G,'minimize',forbidden);monkeypatch.setattr(G.Fitter,'fit',forbidden)
    d,p,x,g=fixture();value,score=G.objective(p,d,G.pattern_parts(x,g,4),G._grid(61))
    assert D.reference_nll(p,d,x,g,61)==pytest.approx(value,abs=1e-12)
    numeric,_=D.finite_difference(p,d,x,g,61,1e-5)
    np.testing.assert_allclose(numeric,score,rtol=1e-6,atol=1e-8)


def test_physical_score_chain_rule_and_normal_quadrature():
    d,p,x,g=fixture();score=np.arange(d.size)/100;physical=D.physical_score(score,p,d)
    jac=np.eye(d.size);blocks={}
    for j,(kind,group,item,c) in enumerate(d.labels):
        if kind=='b':blocks.setdefault((group,item),[]).append(j)
    for indices in blocks.values():
        jac[np.ix_(indices,indices)]=0
        for i,j in enumerate(indices):
            jac[j,indices[0]]=1
            for k in indices[1:i+1]:jac[j,k]=np.exp(p[k])
    jac[-1,-1]=np.exp(p[-1])
    np.testing.assert_allclose(jac.T@physical,score,atol=1e-12)
    for q in D.POINTS:
        z,lw=D.reference_grid(q)
        assert np.exp(lw).sum()==pytest.approx(1.,abs=1e-13)
        assert np.exp(lw)@(z*z)==pytest.approx(1.,abs=1e-13)


def test_original_evidence_roundtrip_and_shared_h2():
    assert D.protect()==146
    r,x,g,path=D.load_saved()
    assert x.shape==(1000,10) and np.bincount(g).tolist()==[500,500]
    solutions=[]
    for item,fits in r['result']['fits'].items():
        for h,f in fits.items():
            d,p,error=D.recover(item,h,f)
            assert error<1e-12
            if h=='H2':solutions.append(p)
    np.testing.assert_array_equal(*solutions)


def test_cached_diagnostic_integrity_and_derivative_checks_no_refit():
    import json
    r=json.loads((D.HERE/'results/diagnostic.json').read_text())
    assert r['fits_executed']==0 and r['prior_files_verified']==D.protect()==146
    assert len(r['slots'])==8 and len(r['unique_solutions'])==7
    for name,digest in r['sources'].items():assert D.sha((D.ROOT/name).read_bytes())==digest
    assert D.sha((D.PRIOR/r['source_archive']).read_bytes())==r['source_archive_sha256']
    assert D.sha((D.PRIOR/r['source_failed_matrix']).read_bytes())==json.loads((D.HERE/'prior_manifest.json').read_text())[str((D.PRIOR/r['source_failed_matrix']).relative_to(D.ROOT))]
    for u in r['unique_solutions'].values():
        assert set(u['evaluations'])=={'61','81','101','121','161'}
        # Arithmetic reproduction tolerances only; NOT scientific gate changes.
        assert abs(u['saved_loglik_reconstruction_error'])<1e-8
        assert abs(u['saved_score_reconstruction_error'])<1e-10
        assert abs(u['saved_audit_score_reconstruction_error'])<1e-10
        for q,e in u['evaluations'].items():
            gradient=np.array(e['normalized_gradient']);j=int(np.argmax(np.abs(gradient)))
            assert e['max_parameter']==u['labels'][j]
            assert e['max_abs_gradient']==abs(gradient[j])
            assert e['native_gate_exceeded']==bool(np.max(np.abs(gradient))>1e-5)
            assert e['total_loglik_score_at_max']==-r['n']*gradient[j]
            assert abs(e['native_minus_independent_loglik'])<1e-8
        for q,checks in u['finite_differences'].items():
            analytic=np.array(u['evaluations'][q]['normalized_gradient'])
            for step,fd in checks.items():
                error=np.abs(analytic-fd['gradient'])
                assert fd['max_abs_error']==error.max()
                np.testing.assert_allclose(fd['gradient'],analytic,atol=1e-8,rtol=1e-5)
    # Diagnose only: never rewrite the prior p suppression or classify it as valid.
    record,_,_,_=D.load_saved()
    assert not record['result']['valid']
    assert all(v['p'] is None for v in record['result']['table'])
