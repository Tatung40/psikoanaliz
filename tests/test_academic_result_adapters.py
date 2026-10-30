from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.academic_result_adapters import (
    irt_report_to_result, dif_report_to_result, gt_report_to_result,
)
from core.academic_result_schema import result_json


@pytest.mark.parametrize("model", ["2PL", "RSM"])
def test_irt_pilot_preserves_observed_values(model):
    report = SimpleNamespace(model=model,
        metadata={"N": 200, "Madde": 2, "Yakınsama": True, "LogLik": -123.4,
                  "Motor": "fixture", "Kestirim": "ML"},
        parameters=pd.DataFrame({"Madde": ["M1", "M2"], "b": [-0.3, 0.7]}),
        curves=pd.DataFrame({"Theta": [0.0], "Probability": [0.5]}),
        information=pd.DataFrame({"Theta": [0.0], "I": [1.2]}),
        warnings=["Scientific gate pending"], errors=[])
    out = irt_report_to_result(report)
    assert out["parameters"][0]["b"] == -0.3
    assert out["sample"] == {"n": 200, "k": 2}
    assert out["standard_errors"] == []
    assert '"schema_version"' in result_json(out)
    report.metadata["Yakınsama"] = False
    with pytest.raises(ValueError, match="Convergence"):
        irt_report_to_result(report)


def test_dif_pilot_never_invents_significance():
    out = dif_report_to_result({
        "metadata": {"Model": "2PL", "N_reference": 120, "N_focal": 110,
                     "Items": ["M1", "M2"], "Inference": False,
                     "Correction_applied": False, "Reference": "'A'", "Focal": "'B'",
                     "Limitations": "NOT CERTIFIED"},
        "tables": {"DIF": pd.DataFrame({"Item": ["M1"], "LR": [2.4],
                                        "p": [np.nan], "q": [np.nan]}),
                   "Parameters": pd.DataFrame({"Item": ["M1"], "Value": [0.8]})}})
    assert out["metadata"]["inference_enabled"] is False
    assert out["tables"]["DIF"][0]["p"] is None
    assert out["sample"]["n"] == 230
    result_json(out)


def test_gt_pilot_preserves_variance_and_decision_coefficients():
    result = {"counts": {"p": 60, "i": 5}, "status": "LIBRARY TESTED",
              "method": "Henderson I", "settings": {"design": "p x i"},
              "anova": pd.DataFrame({"SS": [2.0]}),
              "variance": pd.DataFrame({"Variance": [0.2]}, index=["p"]),
              "g": pd.DataFrame({"rho^2": [0.82]}, index=["p"]),
              "errors": pd.DataFrame({"relative": [0.04]}),
              "d": pd.DataFrame({"i": [5], "rho^2": [0.82], "phi^2": [0.79]})}
    out = gt_report_to_result(result)
    assert out["tables"]["d"][0]["phi^2"] == 0.79
    assert out["parameters"][0]["Variance"] == 0.2
    result_json(out)
