"""Fixed-parameter public-library diagnostics; ZERO model calibrations.

FitResult.from_dict restores a deliberately synthetic native result solely
for API/mathematical tests. This is not independent R or convergence evidence.
"""
import copy
import inspect
from importlib.metadata import version
import numpy as np
import pandas as pd
import pytest
import mirt
from core.irt_mirt_diagnostics import enrich, _directions, IDENTITY_ATOL


@pytest.fixture
def native():
    from mirt.models.multidimensional import MultidimensionalModel
    from mirt.results.fit_result import FitResult
    from core.irt_multidim import surface_tables
    pattern=np.array([[1,0],[1,0],[0,1],[0,1],[1,1],[1,1]],bool)
    names=[f"M{i+1}" for i in range(len(pattern))]
    model=MultidimensionalModel(len(pattern),2,model_type="confirmatory",
                                loading_pattern=pattern,item_names=names)
    slopes=pattern.astype(float)*1.2
    model.set_parameters(slopes=slopes,intercepts=np.linspace(-.4,.4,len(pattern)))
    x=pd.DataFrame(np.array([[((i+j)//(j+1))%2 for j in range(len(pattern))]
                            for i in range(16)],int),columns=names)
    ids=[f"Kişi-{i+1}" for i in range(len(x))]
    covariance=np.array([[1.,.25],[.25,1.]])
    synthetic=FitResult(model=model,log_likelihood=-30.,n_iterations=0,converged=True,
                       standard_errors={},aic=88.,bic=99.,n_observations=len(x),
                       n_parameters=14,latent_covariance=covariance,latent_mean=np.zeros(2))
    fit=FitResult.from_dict(synthetic.to_dict())  # public saved-model restoration; NO FIT
    score=mirt.fscores(fit,x.to_numpy(),method="EAP",n_quadpts=15,person_ids=ids)
    eap=pd.DataFrame(score.theta,columns=["F1","F2"]);eap.insert(0,"ID",ids)
    for j in range(2):eap[f"Posterior_SD_F{j+1}"]=score.standard_error[:,j]
    tables=surface_tables(fit.model,names,("0","1"),2);tables["EAP"]=eap
    outcome=dict(model="2PL",fit=fit,coded=x,pattern=pattern,covariance=covariance,
                 labels=("0","1"),settings=dict(n_quadpts=15),exploratory=False)
    return outcome,dict(tables=tables,metadata={}),ids


def test_installed_public_posterior_contract_and_no_calibration(native,monkeypatch):
    import mirt.scoring as scoring
    assert version("mirt")=="1.2.0"
    assert {"n_quadpts","person_ids","batch_size"} <= set(inspect.signature(scoring.ability_posterior).parameters)
    original=scoring.ability_posterior;calls=[]
    def posterior(*args,**kwargs):
        calls.append(kwargs.copy());return original(*args,**kwargs)
    def forbidden(*args,**kwargs):raise AssertionError("Unexpected model calibration")
    monkeypatch.setattr(scoring,"ability_posterior",posterior)
    monkeypatch.setattr(mirt,"fit_mirt",forbidden)
    outcome,result,ids=native
    before=outcome["fit"].model.parameters
    saved={k:v.copy() for k,v in before.items()}
    out=enrich(outcome,result,ids)
    assert out is result and len(calls)==1 and calls[0]["n_quadpts"]==15
    assert calls[0]["person_ids"]==ids and calls[0]["batch_size"]<=128
    for k,v in saved.items():np.testing.assert_array_equal(before[k],v)
    assert set(result["tables"]["DerivedDiagnostics"]["Status"])=={"DESCRIPTIVE IDENTITY"}
    assert result["metadata"]["Posterior_covariance_independent_R_certification"]=="NOT CERTIFIED"
    assert result["metadata"]["Posterior_covariance_status"]=="DESCRIPTIVE IDENTITY"
    assert "koşullu" in result["metadata"]["Posterior_covariance_reason"]


def test_full_conditional_posterior_covariance_matches_public_grid(native):
    from mirt.scoring import ability_posterior
    outcome,result,ids=native;enrich(outcome,result,ids)
    posterior=ability_posterior(outcome["fit"],outcome["coded"].to_numpy(),
                                n_quadpts=15,person_ids=ids,batch_size=128)
    actual=result["tables"]["PosteriorCovariance"]
    assert actual["ID"].drop_duplicates().tolist()==ids
    matrices=np.stack([actual.loc[actual.ID==pid].pivot(index="Row",columns="Column",values="Covariance").to_numpy()
                       for pid in ids])
    manual=[]
    for w,mu in zip(posterior.weights,posterior.mean):
        delta=posterior.points-mu
        manual.append(delta.T @ (w[:,None]*delta))
    np.testing.assert_allclose(matrices,manual,atol=IDENTITY_ATOL,rtol=0)
    np.testing.assert_allclose(np.diagonal(matrices,axis1=1,axis2=2),
                               posterior.standard_error**2,atol=IDENTITY_ATOL,rtol=0)
    assert np.linalg.eigvalsh(matrices).min()>=-IDENTITY_ATOL


def _matrix_table(matrix,item=False):
    rows=[dict(Node=0,Row=r+1,Column=c+1,Information=matrix[r,c])
          for r in range(len(matrix)) for c in range(len(matrix))]
    if item:
        for row in rows:row["Item"]="M1"
    return pd.DataFrame(rows)


def test_directional_information_and_joint_uncertainty_are_distinct():
    table=_directions(_matrix_table(np.array([[2.,1.],[1.,2.]])),2,False)
    axis=table.loc[table.Direction=="F1"].iloc[0]
    assert axis.Information==2
    assert axis.Direction_fixed_orthogonal_Fisher_SD==pytest.approx(1/np.sqrt(2))
    assert axis.Joint_contrast_Fisher_SD==pytest.approx(np.sqrt(2/3))
    singular=_directions(_matrix_table(np.array([[1.,0.],[0.,0.]]),True),2,True)
    assert singular.Joint_contrast_Fisher_SD.isna().all()
    zero=singular.loc[singular.Direction=="F2"].iloc[0]
    assert zero.Information==0 and np.isnan(zero.Direction_fixed_orthogonal_Fisher_SD) and zero.Note
    with pytest.raises(ValueError,match="pozitif"):
        _directions(_matrix_table(np.array([[1.,0.],[0.,-1.]])),2,False)


def test_directional_native_information_equals_probability_derivative(native):
    outcome,result,ids=native;enrich(outcome,result,ids)
    model=outcome["fit"].model
    theta=np.array([[-3.,-3.]])
    h=1e-5
    p=model.probability(theta)[0]
    table=result["tables"]["Directional_IIF"]
    names=list(outcome["coded"].columns)
    for direction in (np.array([1.,0.]),np.array([0.,1.]),np.ones(2)/np.sqrt(2)):
        derivative=(model.probability(theta+h*direction)[0]-model.probability(theta-h*direction)[0])/(2*h)
        expected=derivative**2/(p*(1-p))
        for j,name in enumerate(names):
            rows=table.loc[(table.Node==0)&(table.Item==name)]
            row=rows.loc[np.isclose(rows.Direction_F1,direction[0]) & np.isclose(rows.Direction_F2,direction[1])].iloc[0]
            assert row.Information==pytest.approx(expected[j],abs=1e-8,rel=1e-6)


def test_loading_correlation_labels_do_not_create_scalar_difficulty(native):
    outcome,result,ids=native;enrich(outcome,result,ids)
    load=result["tables"]["LoadingMatrix"]
    np.testing.assert_array_equal(load[["a1","a2"]],outcome["fit"].model.parameters["slopes"])
    assert "difficulty" not in " ".join(load.columns).lower()
    corr=result["tables"]["Correlation"]
    np.testing.assert_allclose(corr[["F1","F2"]],outcome["covariance"],atol=IDENTITY_ATOL,rtol=0)
    assert "standartlaştırılmış AFA" in result["metadata"]["Derived_diagnostics_note"]


def test_unavailable_api_inconsistent_scores_and_memory_fail_closed(native,monkeypatch):
    import core.irt_mirt_diagnostics as D
    import mirt.scoring as scoring
    outcome,result,ids=native
    original=scoring.ability_posterior
    monkeypatch.setattr(scoring,"ability_posterior",None)
    enrich(outcome,result,ids)
    assert "PosteriorCovariance" not in result["tables"]
    assert result["metadata"]["Posterior_covariance_status"]=="WITHHELD"
    assert "API" in result["metadata"]["Posterior_covariance_reason"]
    status=result["tables"]["DerivedDiagnostics"].set_index("Output")
    assert status.loc["PosteriorCovariance","Status"]=="WITHHELD"
    assert "API" in status.loc["PosteriorCovariance","Scope"]
    monkeypatch.setattr(scoring,"ability_posterior",original)
    result["tables"]["EAP"].loc[0,"F1"]+=.1
    enrich(outcome,result,ids)
    assert "PosteriorCovariance" not in result["tables"]
    result["tables"]["EAP"].loc[0,"F1"]-=.1
    monkeypatch.setattr(D,"MAX_POSTERIOR_CELLS",1)
    enrich(outcome,result,ids)
    assert "PosteriorCovariance" not in result["tables"]
    assert "bellek" in result["tables"]["DerivedDiagnostics"].set_index("Output").loc["PosteriorCovariance","Scope"]


def test_nonconvergence_refuses_new_outputs_and_advanced_untouched(native):
    outcome,result,ids=native
    outcome["fit"].converged=False;enrich(outcome,result,ids)
    assert set(result["tables"]["DerivedDiagnostics"].query("Output != 'Boundaries'").Status)=={"WITHHELD"}
    assert result["metadata"]["Boundary_status"]=="DESCRIPTIVE IDENTITY"
    assert not {"PosteriorCovariance","Directional_IIF","Directional_TIF","Directional_Curves","LoadingMatrix","Correlation"} & set(result["tables"])
    for change in (dict(model="Bifactor"),dict(model="3PL"),dict(exploratory=True)):
        advanced=dict(outcome,**change)
        before=copy.deepcopy(result)
        assert enrich(advanced,result,ids) is result
        assert result["metadata"]==before["metadata"]
        for key in before["tables"]:pd.testing.assert_frame_equal(result["tables"][key],before["tables"][key])


def test_directional_curves_are_native_probabilities_on_unit_vector_paths(native):
    outcome,result,ids=native;enrich(outcome,result,ids)
    curves=result["tables"]["Directional_Curves"]
    q=outcome["pattern"].shape[1]
    assert len(curves)==(q+1)*25*len(outcome["coded"].columns)*len(outcome["labels"])
    assert curves.Direction.drop_duplicates().tolist()==["F1","F2","Birim köşegen"]
    assert curves.Theta.min()==-3 and curves.Theta.max()==3
    grouped=curves.groupby(["Direction","Theta","Item"],sort=False).Probability.sum()
    np.testing.assert_allclose(grouped,1,atol=IDENTITY_ATOL,rtol=0)
    for name,rows in curves.groupby("Direction",sort=False):
        vector=rows[[f"Direction_F{j+1}" for j in range(q)]].iloc[0].to_numpy(float)
        assert np.linalg.norm(vector)==pytest.approx(1)
        if name=="F1":np.testing.assert_array_equal(vector,[1,0])
        elif name=="F2":np.testing.assert_array_equal(vector,[0,1])
        else:np.testing.assert_allclose(vector,np.ones(2)/np.sqrt(2),atol=IDENTITY_ATOL,rtol=0)
        values=rows.Theta.drop_duplicates().to_numpy()
        actual=outcome["fit"].model.probability(values[:,None]*vector[None,:])
        expected=np.stack([1-actual,actual],axis=-1).reshape(-1)
        np.testing.assert_array_equal(rows.Probability.to_numpy(),expected)
    assert result["tables"]["DerivedDiagnostics"].set_index("Output").loc["Directional_Curves","Status"]=="DESCRIPTIVE IDENTITY"


def test_actual_native_2pl_bounds_detect_both_endpoints(native):
    from core.irt_mirt_diagnostics import native_boundaries
    outcome,result,ids=native
    model=outcome["fit"].model
    slopes=model.parameters["slopes"].copy()
    slopes[0,0]=.1;slopes[1,0]=5
    model.set_parameters(slopes=slopes)
    table=native_boundaries(outcome["fit"])
    active=table.loc[table.Parameter=="slopes"]
    assert set(active.Lower)=={.1} and set(active.Upper)=={5.}
    for coordinate in ("(0, 0)","(1, 0)"):
        assert bool(active.set_index("Coordinate").loc[coordinate,"At_bound"])
    enrich(outcome,result,ids)
    assert result["metadata"]["At_box_bound"] is True
    assert result["tables"]["Boundaries"].equals(table)


def test_actual_native_nominal_signed_bounds_and_fixed_unit_exclusion(native):
    from types import SimpleNamespace
    from mirt.models.polytomous import NominalResponseModel
    from core.irt_mirt_diagnostics import native_boundaries
    outcome,_,_=native;pattern=outcome["pattern"]
    nominal=NominalResponseModel(len(pattern),4,n_factors=2)
    mask=np.broadcast_to(pattern[:,None,:],(len(pattern),4,2)).copy();mask[:,0,:]=False
    intercept_mask=np.ones((len(pattern),4),bool);intercept_mask[:,0]=False
    nominal.set_free_parameter_masks(dict(slopes=mask,intercepts=intercept_mask))
    slopes=np.zeros((len(pattern),4,2));slopes[0,1,0]=-5;slopes[1,1,0]=5
    nominal.set_parameters(slopes=slopes,intercepts=np.zeros((len(pattern),4)))
    table=native_boundaries(SimpleNamespace(model=nominal))
    active=table.loc[table.Parameter=="slopes"]
    assert set(active.Lower)=={-5.} and set(active.Upper)=={5.}
    for coordinate in ("(0, 1, 0)","(1, 1, 0)"):
        assert bool(active.set_index("Coordinate").loc[coordinate,"At_bound"])
    model=outcome["fit"].model
    model.set_free_parameter_masks(dict(slopes=np.zeros_like(pattern,bool),
                                         intercepts=np.ones(len(pattern),bool)))
    model.set_parameters(slopes=pattern.astype(float))
    fixed=native_boundaries(outcome["fit"])
    assert set(fixed.Parameter)=={"intercepts"}
