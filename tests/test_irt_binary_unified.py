import numpy as np
import pytest
from core.irt_binary_unified import BINARY_MODELS, fit_binary

@pytest.fixture
def data():
    from scipy.special import expit
    rng=np.random.default_rng(20261022)
    theta=rng.normal(size=350)
    b=np.linspace(-1.5,1.5,12)
    a=np.linspace(.7,1.5,12)
    return (rng.random((350,12))<expit(a[None,:]*(theta[:,None]-b[None,:]))).astype(int)

def test_supported_models_complete():
    assert BINARY_MODELS==("Rasch","1PL","2PL","3PL","4PL")

@pytest.mark.parametrize("model",["Rasch","1PL","2PL"])
def test_native_real_fits(model,data):
    kw={"max_iter":150} if model=="Rasch" else {"max_iter":30}
    out=fit_binary(data,model,**kw)
    assert out.model==model and out.engine=="eser"
    assert out.result is not None

@pytest.mark.parametrize("model",["3PL","4PL"])
def test_native_high_parameter_models(data,model):
    out=fit_binary(data,model,max_iter=10)
    assert out.model==model and out.result is not None

def test_rasch_is_not_mislabeled_as_python_mirt_1pl(data):
    with pytest.raises(ValueError,match="free-latent-SD"):
        fit_binary(data,"Rasch",engine="python_mirt")

def test_no_implicit_engine_fallback(data):
    with pytest.raises(ValueError,match="Unknown engine"):
        fit_binary(data,"2PL",engine="random")
    with pytest.raises(ValueError,match="Unsupported"):
        fit_binary(data,"5PL")

@pytest.mark.parametrize("model",["3PL","4PL"])
def test_optional_python_engine_real_fits(data,model):
    out=fit_binary(data,model,engine="python_mirt")
    assert out.engine=="python_mirt"
    assert out.scientific_status.startswith("EXPERIMENTAL")
    assert out.warnings
