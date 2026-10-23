import pytest
from core.irt_python_mirt_adapter import fit_python_mirt, SUPPORTED

def test_rsm_is_not_advertised_as_public_python_mirt_model():
    assert "RSM" not in SUPPORTED
    with pytest.raises(ValueError, match="RSM"):
        fit_python_mirt([[0,1,2],[2,1,0],[1,2,0]], model="RSM", n_categories=3)
