"""Pre-registered 3PL/4PL Wald CI coverage evidence and the screen decision derived from it (MC not run in CI)."""
import json
from pathlib import Path

import numpy as np
import pytest

from core.irt_binary_inference import ACCURACY_34, BIAS_LIMIT, EVIDENCE_34, parameter_inference
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
            bias, rmse = ACCURACY_34[(r["model"], r["estimation"], q)]
            assert abs(bias) >= abs(r[q]["mean_bias"]) - 5e-4 and rmse >= r[q]["rmse"] - 5e-4
    assert {k for k, v in ACCURACY_34.items() if abs(v[0]) > BIAS_LIMIT and EVIDENCE_34[k][2]} == {("4PL", "ML", "a")}


@pytest.mark.parametrize("model,estimation", [("3PL", "MAP"), ("4PL", "MAP"), ("3PL", "ML"), ("4PL", "ML")])
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
        # Historical coverage decision is necessary, not sufficient: 4PL ML
        # is excluded by the independent parameter-interpretation policy.
        supported = EVIDENCE_34[(model, estimation, row["Parametre"])][2] and (model, estimation) != ("4PL", "ML")
        assert np.isfinite(row["GA %95 alt"]) == (supported and np.isfinite(row["SH"])), row.to_dict()
        biased = supported and abs(ACCURACY_34[(model, estimation, row["Parametre"])][0]) > BIAS_LIMIT
        assert (row["Not"] != "") == (not supported or biased or not np.isfinite(row["SH"]))
        if biased and np.isfinite(row["SH"]):
            assert "sapma +0.179" in row["Not"] and "RMSE 0.504" in row["Not"]


def test_4pl_ml_policy_suppresses_intervals_in_ui_csv_xlsx_and_zip(monkeypatch):
    import io
    import zipfile
    from types import SimpleNamespace
    import pandas as pd
    from core.irt_python_mirt_report import LibraryReport
    from reports import irt_binary_report
    from ui.irt_python_library_view import _csv

    # Interior, positive-definite information: the historical a-coverage gate
    # alone would release all a intervals. No optimizer is needed to test policy.
    co = pd.DataFrame({"discrimination": [1.2]*6, "difficulty": [0.]*6,
                       "guessing": [.15]*6, "upper": [.9]*6}, index=[f"M{i}" for i in range(6)])
    labels = [f"{q}[{item}]" for item in co.index for q in co.columns]
    fit = SimpleNamespace(converged=True, coef=lambda: co, vcov=np.eye(24)*.01, vcov_labels=labels)
    raw, _ = parameter_inference(fit, "4PL", co.index)
    shown, reason = parameter_inference(fit, "4PL", co.index, "ML")
    assert EVIDENCE_34[("4PL", "ML", "a")][2]  # original scientific evidence intact
    assert raw[["GA %95 alt", "GA %95 üst"]].notna().all().all()
    np.testing.assert_array_equal(shown.SH, raw.SH)
    assert shown[["GA %95 alt", "GA %95 üst"]].isna().all().all()
    assert shown.Not.str.contains("4PL ML kullanım politikası").all()
    # UI and all downloads consume this same table; no fresh/raw inference in exports.
    monkeypatch.setattr(irt_binary_report, "png_bytes", lambda *args: b"png")
    report = LibraryReport("4PL", {"N": 1000, "Madde": 6, "Yakınsama": True})
    xlsx, archive = irt_binary_report.bundle(report, pd.DataFrame(), shown, reason)
    exported = [pd.read_csv(io.BytesIO(_csv(shown))),
                pd.read_excel(io.BytesIO(xlsx), sheet_name="Parametre_SH_GA")]
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        exported.append(pd.read_csv(z.open("tablolar/Parametre_SH_GA.csv")))
    for table in exported:
        assert table[["GA %95 alt", "GA %95 üst"]].isna().all().all()
        np.testing.assert_allclose(table.SH, shown.SH)
