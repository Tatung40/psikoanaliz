"""Amendment C3: the Python mirt Rasch screen at the 61-node default (certified fixed-a 1PL) — Wright map outputs equal
the fitted item b and the library EAP of the same fit; with 21 nodes set explicitly the screen still reproduces the
frozen phase-2 evidence (historical reproducibility)."""
import io
from pathlib import Path

import mirt
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "validation/r_mirt_phase2/evidence"


def _screen(monkeypatch, quadpts=None):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    x = pd.read_csv(ROOT / "tests/fixtures/rasch/data_rasch_main.csv").astype(int)
    ids = [f"K{i:04}" for i in range(len(x))]; exports = {}
    original = st.download_button
    def capture(label, data, *args, **kwargs):
        exports[kwargs.get("key")] = data
        return original(label, data, *args, **kwargs)
    monkeypatch.setattr(st, "download_button", capture)
    data = x.copy(); data.insert(0, "ID", ids)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    for k, v in dict(raw_df=data, scored_df=None, item_cols=list(x.columns), id_col="ID", group_col=None,
                     is_dichotomous=True, nav_page="MTK: Python mirt Rasch").items():
        at.session_state[k] = v
    at.run()
    if quadpts is not None:
        at.checkbox(key="python_mirt_solver_advanced_Rasch").check().run()
        at.number_input(key="python_mirt_solver_Rasch_n_quadpts").set_value(quadpts).run()
    at.button(key="python_mirt_start_Rasch").click().run()
    assert not at.exception
    read = lambda key: pd.read_csv(io.BytesIO(exports[key]))
    return x, at, read("rasch_wright_persons"), read("rasch_wright_items"), read("python_mirt_persons_csv_Rasch")


def test_default_61_nodes_wright_outputs_are_the_fit_and_library_eap(monkeypatch):
    from core.irt_python_mirt_adapter import fit_python_mirt
    x, at, wp, wi, persons = _screen(monkeypatch)
    saved = at.session_state["python_mirt_fit_Rasch"]["outcome"]
    assert saved.solver_settings["n_quadpts"] == 61
    ref = fit_python_mirt(x.to_numpy(), model="Rasch", n_quadpts=61).fit                       # independent refit
    np.testing.assert_allclose(wi.set_index("Madde").loc[x.columns, "b"], ref.coef().difficulty.to_numpy(float), rtol=0, atol=1e-12)
    eap = mirt.fscores(ref, x.to_numpy(), method="EAP", n_quadpts=61)
    np.testing.assert_allclose(wp["θ"], np.asarray(eap.theta).reshape(-1), rtol=0, atol=1e-12)
    np.testing.assert_allclose(persons["Yetenek (θ)"], wp["θ"], rtol=0, atol=1e-14)
    from core.irt_binary_usage_policy import certification
    assert certification("Rasch", "ML", len(x), x.shape[1], saved.solver_settings)[0] == "outside_scope"   # k = 10 < 12


def test_explicit_21_nodes_reproduce_frozen_phase2_evidence(monkeypatch):
    x, at, wp, wi, persons = _screen(monkeypatch, quadpts=21)
    ep = pd.read_csv(FROZEN / "rasch_persons.csv"); ei = pd.read_csv(FROZEN / "rasch_items.csv")
    np.testing.assert_allclose(wp["θ"], ep["θ"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(wp["Posterior SD"], ep["Posterior SD"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(wi.set_index("Madde").loc[ei.Madde, "b"], ei.b, rtol=1e-12, atol=1e-12)
