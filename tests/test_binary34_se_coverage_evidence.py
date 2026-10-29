"""Pre-registered 3PL/4PL Wald CI coverage evidence and the screen decision derived from it (MC not run in CI)."""
import json
from pathlib import Path

import numpy as np
import pytest

from core.irt_binary_inference import EVIDENCE_34, parameter_inference
from validation.binary34_se_coverage_study import CRITERIA, DESIGN, judge

EVIDENCE = Path(__file__).resolve().parents[1] / "validation" / "evidence" / "p7_binary34_se" / "coverage.json"


def test_stored_evidence_matches_design_and_the_screen_decision_table():
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["design"] == json.loads(json.dumps(DESIGN)) and evidence["criteria"] == CRITERIA
    checks, supported = judge(evidence["results"])
    assert evidence["checks"] == checks and evidence["wald_ci_supported"] == supported
    assert {f"{m}_{e}_{q}": v[2] for (m, e, q), v in EVIDENCE_34.items()} == supported
    for r in evidence["results"]:
        assert r["converged"] == r["replications"] and r["se_refused"] == 0
        for q, (coverage, withheld, _) in ((q, EVIDENCE_34[(r["model"], r["estimation"], q)]) for q in "abcd" if q in r):
            assert abs(coverage - .95) >= abs(r[q]["coverage"] - .95) - 5e-4 and withheld >= r[q]["withheld_rate"] - 5e-4


@pytest.mark.parametrize("model,estimation", [("3PL", "MAP"), ("4PL", "MAP"), ("3PL", "ML")])
def test_screen_keeps_se_and_withholds_only_unsupported_intervals(model, estimation):
    pytest.importorskip("mirt")
    from core.irt_python_mirt_adapter import fit_python_mirt
    from validation.irt_3pl4pl_root_cause import simulate
    x, _ = simulate(model, 1000, 12, 3)
    fit = fit_python_mirt(x, model=model, estimation=estimation).fit
    names = [f"M{j}" for j in range(x.shape[1])]
    raw, _ = parameter_inference(fit, model, names)
    shown, _ = parameter_inference(fit, model, names, estimation)
    np.testing.assert_array_equal(raw.SH.to_numpy(), shown.SH.to_numpy())
    for _, row in shown.iterrows():
        supported = EVIDENCE_34[(model, estimation, row["Parametre"])][2]
        assert np.isfinite(row["GA %95 alt"]) == (supported and np.isfinite(row["SH"])), row.to_dict()
        assert (row["Not"] != "") == (not supported or not np.isfinite(row["SH"]))
