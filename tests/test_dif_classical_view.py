"""Observed-score DIF cache provenance and export suppression; no simulations."""
import io
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from streamlit.delta_generator import DeltaGenerator
from ui import dif_classical_view as U


def data():
    x = pd.DataFrame(np.random.default_rng(9).integers(0, 2, (30, 4)), columns=list("ABCD"))
    x["group"] = ["R", "F", "X"] * 10
    x["other_group"] = ["X", "R", "F"] * 10
    return x


def app():
    at = AppTest.from_string('''
import streamlit as st
from ui.dif_classical_view import render
render(st.session_state.data, list("ABCD"), group_col="group")
''')
    at.session_state["data"] = data()
    return at


@pytest.mark.parametrize("change", ["values", "ordinal", "rows", "group_values", "group", "reference",
    "focal", "items", "too_few_items", "one_group", "method", "adjust", "alpha", "purify", "max_rounds"])
def test_changes_clear_results_and_both_downloads(change, monkeypatch):
    calls, exports = [], []
    def fit(x, g, ref, foc, **kw):
        calls.append(kw)
        return pd.DataFrame({"Madde": list(x), "DIF (düzeltilmiş p < alpha)": [True] * len(x.columns),
            "delta": 1., "se_log_or": .1, "delta_r2": .04}), dict(n_reference=10, n_focal=10,
            adjust=kw["adjust"], ordinal=kw["ordinal"], rounds=1, notes=[])
    monkeypatch.setattr(U, "dif_screen", fit)
    original = DeltaGenerator.download_button
    def download(self, label, payload, *args, **kwargs):
        exports.append(payload)
        return original(self, label, payload, *args, **kwargs)
    monkeypatch.setattr(DeltaGenerator, "download_button", download)
    at = app().run()
    at.button(key="cdif_run").click().run()
    assert not at.exception and len(calls) == 1 and len(exports) == 2
    assert len(at.dataframe) == 1 and at.metric and at.get("plotly_chart")
    assert pd.read_csv(io.BytesIO(exports[0])).shape[0] == 4
    assert pd.read_excel(io.BytesIO(exports[1]), sheet_name="DIF").shape[0] == 4
    saved_key = at.session_state["cdif_result"]["key"]
    at.run()  # Same values/settings retain valid results without recalibrating.
    assert not at.exception and len(calls) == 1 and len(exports) == 4
    assert at.session_state["cdif_result"]["key"] == saved_key
    changed = data()
    if change in ("values", "ordinal", "rows", "group_values", "one_group"):
        if change == "values": changed.loc[0, "A"] = 1 - changed.loc[0, "A"]
        if change == "ordinal": changed.loc[0, "A"] = 2
        if change == "rows": changed = changed.iloc[::-1].reset_index(drop=True)
        if change == "group_values": changed.loc[0, "group"] = "F"
        if change == "one_group": changed["group"] = "R"
        assert changed.shape == data().shape and changed.columns.equals(data().columns)
        at.session_state["data"] = changed
    elif change == "group": at.selectbox(key="cdif_group").set_value("other_group")
    elif change == "reference": at.selectbox(key="cdif_ref").set_value("X")
    elif change == "focal": at.selectbox(key="cdif_foc").set_value("X")
    elif change == "items": at.multiselect(key="cdif_items").set_value(list("ABC"))
    elif change == "too_few_items": at.multiselect(key="cdif_items").set_value(list("AB"))
    elif change == "method": at.radio(key="cdif_method").set_value("LR")
    elif change == "adjust": at.selectbox(key="cdif_adj").set_value("Holm")
    elif change == "alpha": at.select_slider(key="cdif_alpha").set_value(.01)
    elif change == "purify": at.checkbox(key="cdif_purify").uncheck()
    else: monkeypatch.setattr(U, "_MAX_ROUNDS", 11)
    at.run()
    assert not at.exception and "cdif_result" not in at.session_state
    assert len(calls) == 1 and len(exports) == 4
    assert not at.dataframe and not at.metric
    assert not at.get("plotly_chart") and not at.get("download_button")
    # Restoring the original data must not resurrect a deleted result.
    at.session_state["data"] = data()
    at.run()
    assert not at.exception and "cdif_result" not in at.session_state and len(exports) == 4


def test_legacy_tuple_is_discarded():
    at = app()
    at.session_state["cdif_result"] = (pd.DataFrame(), {}, tuple("ABCD"), "group", "R", "F")
    at.run()
    assert not at.exception and "cdif_result" not in at.session_state
    assert not at.dataframe and not at.get("download_button")
