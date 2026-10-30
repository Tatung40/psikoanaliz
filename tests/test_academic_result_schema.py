import json

import numpy as np
import pandas as pd
import pytest

from core.academic_result_schema import build_result, result_json, validate_result


@pytest.mark.parametrize("family,model", [("CTT", "Classical Item Analysis"), ("IRT", "RSM"), ("GT", "p x i")])
def test_result_roundtrip(family, model):
    result = build_result(family=family, model=model, n=120, k=8,
                          parameters=[{"item": "M1", "estimate": np.float64(0.42)}],
                          standard_errors=[], fit_statistics={"loglik": -12.5},
                          tables={"items": pd.DataFrame({"item": ["M1"], "score": [1]})},
                          warnings=["Illustrative values only"],
                          provenance={"engine": "test-fixture", "validated": False})
    loaded = json.loads(result_json(result))
    assert loaded["family"] == family
    assert loaded["tables"]["items"][0]["item"] == "M1"
    assert loaded["parameters"][0]["estimate"] == 0.42


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_reject_nonfinite_values(bad):
    with pytest.raises(ValueError, match="Non-finite"):
        build_result(family="IRT", model="RSM", n=10, fit_statistics={"loglik": bad})


def test_reject_unknown_model_and_invalid_sample():
    with pytest.raises(ValueError):
        build_result(family="SEM", model="RSM", n=10)
    with pytest.raises(ValueError):
        build_result(family="IRT", model="RSM", n=0)
    with pytest.raises(ValueError):
        build_result(family="IRT", model="", n=10)


def test_reject_missing_fields_and_wrong_types():
    with pytest.raises(ValueError):
        validate_result({"schema_version": "1.0.0"})
    result = build_result(family="GT", model="p x r", n=20)
    result["warnings"] = "not-a-list"
    with pytest.raises(ValueError):
        validate_result(result)
