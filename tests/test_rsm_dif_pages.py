"""Streamlit AppTest: RSM page and observed-score DIF page run end to end."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import irt_rsm as R

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _app(df, page, items, group=None):
    st = pytest.importorskip("streamlit.testing.v1")
    at = st.AppTest.from_file(APP, default_timeout=240)
    for key, value in (("raw_df", df), ("scored_df", df.copy()), ("item_cols", items), ("id_col", None),
                       ("group_col", group), ("is_dichotomous", False), ("nav_page", page)):
        at.session_state[key] = value
    return at.run()


def test_rsm_page_runs_and_shows_parameters():
    x = R.simulate(np.linspace(-1, 1, 6), np.array([-1., 0., 1.]), 400, 1.1, seed=3)
    df = pd.DataFrame(x, columns=[f"L{j}" for j in range(6)])
    at = _app(df, "MTK: RSM", list(df.columns))
    assert not at.exception
    at.button(key="rsm_run").click().run()
    assert not at.exception and at.session_state["rsm_request"]["cols"] == list(df.columns)
    assert len(at.dataframe) >= 6 and not at.error


def test_classical_dif_page_runs():
    rng = np.random.default_rng(4); n = 600
    g = rng.integers(0, 2, n); th = rng.standard_normal(n)
    x = (rng.random((n, 8)) < 1 / (1 + np.exp(-(th[:, None] - np.linspace(-1, 1, 8))))).astype(int)
    df = pd.DataFrame(x, columns=[f"M{j}" for j in range(8)]); df["grup"] = np.where(g, "B", "A")
    at = _app(df, "MTK: Gözlenen Puan DIF", [f"M{j}" for j in range(8)], group="grup")
    assert not at.exception
    at.button(key="cdif_run").click().run()
    assert not at.exception and at.session_state["cdif_result"][0].shape[0] == 8


def test_dif34_bootstrap_page_respects_size_study_gate():
    from scipy.special import expit
    rng = np.random.default_rng(5); n = 300
    g = np.r_[np.zeros(n, int), np.ones(n, int)]; th = rng.standard_normal(2 * n)
    c = np.linspace(.05, .2, 6)
    x = (rng.random((2 * n, 6)) < c + (1 - c) * expit(th[:, None] - np.linspace(-1, 1, 6))).astype(int)
    df = pd.DataFrame(x, columns=[f"M{j}" for j in range(6)]); df["grup"] = np.where(g, "B", "A")
    at = _app(df, "MTK: 3PL/4PL Bootstrap DIF", [f"M{j}" for j in range(6)], group="grup")
    assert not at.exception
    at.select_slider(key="b34_B").set_value(99).run()
    at.button(key="b34_run").click().run()
    assert not at.exception
    saved = at.session_state["b34_result"]
    res, model = saved["result"], saved["model"]
    shown = at.dataframe[0].value["Bootstrap p"].iloc[0]
    from ui.dif34_bootstrap_view import inference_eligibility
    assert (np.isfinite(shown)) == (inference_eligibility(model)[0] and np.isfinite(res["p"]))
