"""Nine named routes, existing fixtures and supported library quantities only."""
import io,json,hashlib
from copy import copy
import numpy as np
import pandas as pd
import pytest
import mirt
from core.irt_python_mirt_adapter import fit_python_mirt
from core.irt_library_diagnostics import library_fit_diagnostics
from reports.irt_b14_report import safe_frame

MODELS=('Rasch','1PL','2PL','3PL','4PL','GRM','PCM','GPCM','NRM')
BINARY=MODELS[:5]


@pytest.mark.parametrize('model',MODELS)
def test_real_nine_model_regression(model,record_property):
    from pathlib import Path
    path=(f'tests/fixtures/irt_b11/data_irt{model.lower()}_ok.csv' if model in ('3PL','4PL') else
        'tests/fixtures/irt_b10/data_irt1pl_ok.csv') if model in BINARY else 'validation/evidence/p5_browser/poly_ordinal.csv'
    x=pd.read_csv(path).iloc[:300,:6].astype(int)
    estimation='MAP' if model in ('3PL','4PL') else 'ML'
    out=fit_python_mirt(x,model=model,n_categories=None if model in BINARY else 4,estimation=estimation)
    fit=out.fit
    assert fit.converged,'Actual fixture fit did not converge; never label as a successful fit'
    diag=library_fit_diagnostics(fit)
    assert diag['converged'] is True and np.isfinite([fit.log_likelihood,fit.aic,fit.bic]).all()
    assert out.solver_settings==dict(max_iter=500,tol=1e-4,n_quadpts=21)
    npar=dict(Rasch=6,**{'1PL':6,'2PL':12,'3PL':18,'4PL':24,'GRM':24,'PCM':18,'GPCM':24,'NRM':36})
    assert fit.n_parameters==npar[model]
    coef=fit.coef();assert isinstance(coef,pd.DataFrame) and len(coef)==6
    assert all(np.isfinite(np.asarray(v,float)).all() for v in fit.model.parameters.values())
    if model in ('Rasch','1PL'):np.testing.assert_array_equal(coef.discrimination,np.ones(6))
    theta=np.linspace(-3,3,31);h=1e-5
    P=np.asarray(fit.model.probability(theta[:,None]))
    assert P.shape==((31,6) if model in BINARY else (31,6,4))
    assert np.isfinite(P).all() and ((P>=0)&(P<=1)).all()
    if model not in BINARY:np.testing.assert_allclose(P.sum(2),1,atol=1e-14)
    derivative=(fit.model.probability((theta+h)[:,None])-fit.model.probability((theta-h)[:,None]))/(2*h)
    independent=derivative**2/(P*(1-P)) if model in BINARY else (derivative**2/P).sum(2)
    I=np.column_stack([fit.model.information(theta[:,None],item_idx=i) for i in range(6)])
    np.testing.assert_allclose(I,independent,rtol=1e-6,atol=1e-8)
    TIF=I.sum(1);assert np.isfinite(1/np.sqrt(TIF)).all() and (TIF>0).all()
    ids=[f'P{i}' for i in range(len(x))]
    sc=mirt.fscores(fit,x.to_numpy(),method='EAP',n_quadpts=61,person_ids=ids)
    assert list(sc.person_ids)==ids
    assert np.asarray(sc.theta).shape[0]==len(x) and np.asarray(sc.standard_error).shape==np.asarray(sc.theta).shape
    assert np.isfinite(sc.theta).all() and np.isfinite(sc.standard_error).all() and (np.asarray(sc.standard_error)>0).all()
    frame=safe_frame(coef.reset_index())
    csv=pd.read_csv(io.StringIO(frame.to_csv(index=False)));buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine='openpyxl') as writer:frame.to_excel(writer,index=False)
    excel=pd.read_excel(io.BytesIO(buf.getvalue()))
    pd.testing.assert_frame_equal(csv,frame,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
    pd.testing.assert_frame_equal(excel,frame,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
    failed=copy(fit);failed.converged=False
    assert library_fit_diagnostics(failed)['converged'] is False
    record_property('actual_model',json.dumps(dict(model=model,estimation=estimation,fixture=path,
        fixture_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),rows=300,items=6,
        converged=bool(fit.converged),iterations=fit.n_iterations,loglik=float(fit.log_likelihood),
        AIC=float(fit.aic),BIC=float(fit.bic),n_parameters=fit.n_parameters,max_information_identity_error=float(abs(I-independent).max()))))
