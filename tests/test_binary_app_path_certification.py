"""Application path after amendment D: 4PL screens default to the D4-validated solver (tol 1e-10, max_iter 50000), show
the certified label in scope, and their IRF/information outputs equal the closed form; 3PL SE/CI stay NOT CERTIFIED."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mirt")
from streamlit.testing.v1 import AppTest

from validation.binary_certification.common import simulate
from validation.binary_certification.g5_identity import closed_form

ROOT = Path(__file__).resolve().parents[1]


def _app(model, page, n=3000, k=12, seed=11):
    x = pd.DataFrame(simulate(model, n, k, seed), columns=[f"M{j + 1}" for j in range(k)])
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    for key, v in dict(raw_df=x, scored_df=x.copy(), item_cols=list(x.columns), id_col=None, group_col=None,
                       is_dichotomous=True, nav_page=page).items():
        at.session_state[key] = v
    at.run()
    return x, at


def test_library_4pl_default_solver_label_and_output_identity():
    from core.irt_python_mirt_report import build_report
    x, at = _app("4PL", "MTK: Python mirt 4PL")
    assert at.radio(key="python_mirt_estimation_4PL").value == "MAP"
    at.button(key="python_mirt_start_4PL").click().run()
    assert not at.exception
    outcome = at.session_state["python_mirt_fit_4PL"]["outcome"]
    assert outcome.estimation == "MAP" and outcome.solver_settings == dict(max_iter=50000, tol=1e-10, n_quadpts=61)
    report = build_report(outcome, model="4PL", items=list(x.columns))
    label = report.metadata["Sertifika durumu"]
    assert label.startswith("CERTIFIED") and "tol ≤ 1e-10" in label
    co = outcome.fit.coef(); t = np.sort(report.curves.Theta.unique())
    P, I = closed_form(*(co[q].to_numpy(float) for q in ("discrimination", "difficulty", "guessing", "upper")), t)
    cur = report.curves[report.curves.Kategori.astype(str) == "1"].pivot(index="Theta", columns="Madde", values="Olasılık")[list(x.columns)]
    assert np.abs(cur.to_numpy() - P).max() <= 1e-10
    assert np.abs(report.information.set_index("θ").loc[t, [f"I_{c}" for c in x.columns]].to_numpy() - I).max() <= 1e-10


def test_binary_route_4pl_default_solver_and_banner():
    x, at = _app("4PL", "MTK: 4PL")
    assert any("4PL: CERTIFIED" in str(i.value) and "tol ≤ 1e-10" in str(i.value) for i in at.info)
    assert at.radio(key="py_mirt_estimation_4PL").value == "MAP"
    at.button(key="py_mirt_run_4PL").click().run()
    assert not at.exception
    fit = at.session_state["python_mirt_experimental_results"]["4PL"]["fit"]
    assert fit.solver_settings == dict(max_iter=50000, tol=1e-10, n_quadpts=61)


def test_other_models_keep_their_defaults_and_3pl_inference_not_certified():
    from core.irt_binary_usage_policy import certification, evidence_note
    ok = dict(max_iter=500, tol=1e-4, n_quadpts=61)
    assert certification("2PL", "ML", 3000, 20, ok)[0] == "certified"
    state, text = certification("3PL", "MAP", 3000, 20, ok)
    assert state == "certified" and text.startswith("OUTPUT-LEVEL CERTIFIED") and "SH ve Wald GA: NOT CERTIFIED" in text
    assert certification("4PL", "MAP", 3000, 20, ok)[0] == "outside_scope"          # restricted to the D4 setting
    assert certification("4PL", "MAP", 3000, 20, dict(max_iter=50000, tol=1e-10, n_quadpts=61))[0] == "certified"
    assert "tol ≤ 1e-10" in evidence_note("4PL")[1]


def test_4pl_ml_keeps_standard_precision():
    from ui.irt_solver_controls import effective_solver
    d4 = dict(max_iter=50000, tol=1e-10, n_quadpts=61)
    assert effective_solver("4PL", "ML", d4) == dict(max_iter=500, tol=1e-4, n_quadpts=61)
    assert effective_solver("4PL", "MAP", d4) == d4 and effective_solver("3PL", "ML", d4) == d4
