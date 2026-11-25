"""Lightweight F-pilot bookkeeping tests; no fits or Monte Carlo."""
import numpy as np
import pytest
from validation.uncertainty_f.runner import coverage_statistics,digest,PROTOCOL_COMMIT


def test_individual_coverage_and_conditional_mcse():
    summary,covered=coverage_statistics([-1,0,1,2],[0,0,0,0],[1,1,1,1])
    assert covered.tolist()==[True,True,True,False]
    assert summary["n"]==4 and summary["covered"]==3
    assert summary["coverage"]==.75
    assert summary["bias"]==-.5
    assert summary["rmse"]==pytest.approx(np.sqrt(1.5))
    assert summary["mcse_conditional_on_calibration"]==pytest.approx(np.sqrt(.75*.25/4))
    assert 0<summary["cp_97_5"][0]<.75<summary["cp_97_5"][1]<1
    assert summary["coverage_screen_only"] is False


@pytest.mark.parametrize("theta,estimate,sd",[
    ([],[],[]),([0],[0,1],[1]),([0],[float("nan")],[1]),([0],[0],[0])])
def test_invalid_rows_fail_without_silent_exclusion(theta,estimate,sd):
    with pytest.raises(ValueError):coverage_statistics(theta,estimate,sd)


def test_hash_covers_values_shape_dtype_and_preregistration():
    a=np.array([[0,1],[1,0]],dtype=np.int64)
    assert digest(a)==digest(a.copy())
    assert digest(a)!=digest(1-a)
    assert digest(a)!=digest(a.reshape(1,4))
    assert digest(a)!=digest(a.astype(float))
    assert PROTOCOL_COMMIT=="72d1d651aa2433fc416cbd960dc7671da9c31e2f"
