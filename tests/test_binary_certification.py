"""Binary IRT certification (validation/binary_certification): decision reproducible from committed evidence, scope
enforcement in the application, equality-constrained SE, MAP SE default unchanged, 61-node binary default."""
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_certification_file_is_the_decision_of_the_committed_evidence():
    from validation.binary_certification.certify import OUT, decide
    assert json.loads(OUT.read_text()) == decide()


def test_status_and_scope_enforcement():
    from core.irt_binary_usage_policy import certification
    decision = json.loads((ROOT / "validation/evidence/binary_certification/CERTIFICATION.json").read_text())["models"]
    good = dict(max_iter=500, tol=1e-4, n_quadpts=61)
    for model, (est, n, k) in {"1PL": ("ML", 1000, 12), "Rasch": ("ML", 10000, 30), "2PL": ("ML", 5000, 20),
                               "3PL": ("MAP", 3000, 12), "4PL": ("MAP", 10000, 30)}.items():
        name = "1PL" if model == "Rasch" else model
        state, text = certification(model, est, n, k, good)
        if decision[name]["status"] in ("CERTIFIED", "OUTPUT-LEVEL CERTIFIED"):
            assert state == "certified" and text.startswith(decision[name]["status"])
            if decision[name]["status"] == "OUTPUT-LEVEL CERTIFIED":
                assert "SH ve Wald GA: NOT CERTIFIED" in text
            for bad in ((est, n, 11), (est, 999 if name in ("1PL", "2PL") else 2999, k), (est, 10001, k),
                        ("ML" if est == "MAP" else "MAP", n, k)):
                assert certification(model, *bad, good)[0] == "outside_scope"
            assert certification(model, est, n, k, dict(good, n_quadpts=21))[0] == "outside_scope"
        else:
            assert state == "not_certified" and "NOT CERTIFIED" in text


def test_common_discrimination_se_is_reported_and_matches_independent_hessian():
    from core.irt_binary_inference import parameter_inference
    from core.irt_python_mirt_adapter import fit_python_mirt
    from validation.binary_certification.common import Objective, simulate
    x = simulate("1PL", 1000, 12, 7, "common_a")
    fit = fit_python_mirt(x, model="1PL", common_discrimination=True, n_quadpts=61).fit
    table, why = parameter_inference(fit, "1PL", [f"M{j}" for j in range(12)])
    assert why is None
    a = table[table.Parametre == "a"]; b = table[table.Parametre == "b"]
    assert len(a) == 12 and a.SH.nunique() == 1 and np.isfinite(a.SH).all()
    coef = fit.coef(); p = {"a": np.full(12, float(coef.discrimination.iloc[0])), "b": coef.difficulty.to_numpy(float)}
    ref = Objective(x, "1PL", points=61).se(p, [("a", None)] + [("b", j) for j in range(12)])
    np.testing.assert_allclose(np.r_[a.SH.iloc[0], b.SH.to_numpy()], ref, rtol=2e-3)


def test_map_se_default_is_the_library_posterior_covariance():
    from core.irt_binary_inference import parameter_inference
    from core.irt_python_mirt_adapter import fit_python_mirt
    from validation.binary_certification.common import simulate
    fit = fit_python_mirt(simulate("3PL", 3000, 12, 11), model="3PL", estimation="MAP", n_quadpts=61).fit
    table, _ = parameter_inference(fit, "3PL", [f"M{j}" for j in range(12)])
    V = np.asarray(fit.vcov() if callable(fit.vcov) else fit.vcov)
    np.testing.assert_allclose(np.sort(table.SH.to_numpy()), np.sort(np.sqrt(np.diag(V))), rtol=1e-12)


def test_binary_screens_default_to_61_nodes_and_other_families_keep_21():
    src = (ROOT / "ui/irt_python_library_view.py").read_text()
    assert "n_quadpts=61 if model in BINARY else 21" in src
    assert 'solver_controls(f"py_mirt_solver_{model}", n_quadpts=61)' in (ROOT / "app.py").read_text()
    assert "solver_controls('pmd_solver')" in (ROOT / "ui/python_mirt_dif_view.py").read_text()


def test_protocol_registered_before_evidence_and_amendments_are_ordered():
    protocol = json.loads((ROOT / "validation/binary_certification/PROTOCOL.json").read_text())
    assert [k for k in protocol if k.startswith("amendment_")] == ["amendment_A1", "amendment_A2", "amendment_A3", "amendment_A4", "amendment_A5", "amendment_B", "amendment_C", "amendment_D"]
    assert protocol["status_rules"]["NOT CERTIFIED"].endswith("the scope is not narrowed after seeing results")


def test_amendment_c_decisions_follow_registered_rules():
    decision = json.loads((ROOT / "validation/evidence/binary_certification/CERTIFICATION.json").read_text())["models"]
    g5 = json.loads((ROOT / "validation/evidence/binary_certification/g5_identity.json").read_text())
    assert all(c["pass"] and max(c["max_abs_diff"].values()) <= 1e-10 for c in g5["cases"])
    three = decision["3PL"]
    assert three["failed_gates"] == ["G3_inference"] and three["status"] == "OUTPUT-LEVEL CERTIFIED"
    assert any("SE and Wald CI" in o for o in three["not_certified_outputs"])
    d4 = json.loads((ROOT / "validation/evidence/binary_certification/d4_4pl.json").read_text())
    g1 = decision["4PL"]["gates"]["G1_R_equivalence"]
    assert g1["pass"] == (g1["expected_cases"] == len(g1["cases"]) and all(g1["cases"].values())) or g1["pass"] == d4["all_pass"]
    assert (decision["4PL"]["status"] == "CERTIFIED") == g1["pass"]


def test_in_scope_rasch_sample_is_independently_validated_and_historical_fixture_kept():
    d3 = json.loads((ROOT / "validation/evidence/binary_certification/d3_rasch_sample.json").read_text())
    import hashlib
    assert d3["pass"] and hashlib.sha256((ROOT / d3["sample"]).read_bytes()).hexdigest() == d3["sha256"] and d3["shape"] == [2000, 20]
    assert (ROOT / "tests/fixtures/rasch/data_rasch_main.csv").exists()
