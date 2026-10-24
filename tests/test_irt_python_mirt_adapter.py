"""Python mirt adapter: deterministic baseline fits and fail-closed inputs."""
import numpy as np
import pytest
from core.irt_python_mirt_adapter import fit_python_mirt

def test_2pl_real_fit():
    import mirt
    x=mirt.simdata(model="2PL",n_persons=100,n_items=5,seed=124)
    r=fit_python_mirt(x,model="2PL")
    assert r.engine=="python_mirt" and r.n_items==5
    assert r.scientific_status.startswith("EXPERIMENTAL")
    assert r.fit is not None

def test_grm_real_fit():
    import mirt
    x=mirt.simdata(model="GRM",n_persons=100,n_items=5,n_categories=4,seed=124)
    r=fit_python_mirt(x,model="GRM",n_categories=4)
    assert r.n_categories==4 and r.fit is not None

@pytest.mark.parametrize("model,categories",[
    ("2PL",np.full((12,5),2)),
    ("GRM",np.full((12,5),5)),
])
def test_invalid_responses(model,categories):
    with pytest.raises(ValueError):
        fit_python_mirt(categories,model=model,n_categories=4 if model=="GRM" else None)

def test_rejects_missing_and_unknown():
    x=np.zeros((10,5));x[0,0]=np.nan
    with pytest.raises(ValueError,match="Missing"):
        fit_python_mirt(x,model="2PL")
    with pytest.raises(ValueError,match="Unsupported"):
        fit_python_mirt(np.zeros((10,5)),model="UNKNOWN")

@pytest.mark.parametrize("model", ["1PL", "2PL", "3PL", "4PL"])
def test_constant_item_is_rejected_before_fit(model):
    x=np.random.default_rng(2026).integers(0,2,size=(80,8))
    x[:,3]=1
    with pytest.raises(ValueError, match="değişkenliği olmayan"):
        fit_python_mirt(x,model=model)

def test_rsm_is_not_silently_routed_to_native():
    with pytest.raises(ValueError, match="RSM"):
        fit_python_mirt(np.zeros((20,5)),model="RSM",n_categories=4)


@pytest.mark.parametrize("model", ["GRM", "PCM", "GPCM", "NRM"])
def test_polytomous_real_fit_smoke(model):
    import mirt
    x = mirt.simdata(model=model, n_persons=120, n_items=5, n_categories=4, seed=2026)
    result = fit_python_mirt(x, model=model, n_categories=4)
    assert result.model == model
    assert result.n_categories == 4
    assert result.fit is not None
    assert result.scientific_status.startswith("EXPERIMENTAL")
