"""UI provenance and fail-closed inference; no Monte Carlo runs."""
import io
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from ui import dif34_bootstrap_view as U


def frame():
    x = pd.DataFrame(np.random.default_rng(4).integers(0, 2, (30, 4)), columns=list("ABCD"))
    x["group"] = ["R", "F", "X"] * 10
    x["other_group"] = ["X", "R", "F"] * 10
    return x


def result(model="4PL", item="A"):
    return dict(item=item, model=model, T_obs=7., p=.001, B_ok=199, B_failed=0,
                boundary_null=True, boundary_alt=True, null_stats=[1., 2., 3.], status="Tamamlandı")


@pytest.mark.parametrize("change", ["values", "rows", "group_values", "group", "reference", "focal",
                                  "item", "anchors", "model", "B", "items"])
def test_stale_results_are_deleted_and_never_exported(change, monkeypatch):
    calls, exports = [], []
    def fit(data, groups, ref, foc, item, anchors, **kw):
        calls.append(1)
        return result(kw["model"], item)
    monkeypatch.setattr(U, "bootstrap_dif_test", fit)
    original = U.st.download_button
    def download(*args, **kwargs):
        exports.append(args[1])
        return original(*args, **kwargs)
    monkeypatch.setattr(U.st, "download_button", download)
    at = AppTest.from_string('''
import streamlit as st
from ui.dif34_bootstrap_view import render
render(st.session_state.data, st.session_state["items"], group_col="group")
''')
    at.session_state["data"] = frame()
    at.session_state["items"] = list("ABCD")
    at.run(); at.button(key="b34_run").click().run()
    assert not at.exception and len(calls) == 1 and len(exports) == 1
    # Identical inputs reuse results without any model recalibration.
    at.run()
    assert not at.exception and len(calls) == 1 and len(exports) == 2
    if change in ("values", "rows", "group_values"):
        changed = frame()
        if change == "values": changed.loc[0, "A"] = 1 - changed.loc[0, "A"]
        if change == "rows": changed = changed.iloc[::-1].reset_index(drop=True)
        if change == "group_values": changed.loc[0, "group"] = "F"
        assert changed.shape == frame().shape
        at.session_state["data"] = changed
    elif change == "group": at.selectbox(key="b34_group").set_value("other_group")
    elif change == "reference": at.selectbox(key="b34_ref").set_value("X")
    elif change == "focal": at.selectbox(key="b34_foc").set_value("X")
    elif change == "item": at.selectbox(key="b34_item").set_value("B")
    elif change == "anchors": at.multiselect(key="b34_anchors").set_value(["B", "C"])
    elif change == "model": at.radio(key="b34_model").set_value("4PL")
    elif change == "B": at.select_slider(key="b34_B").set_value(499)
    else: at.session_state["items"] = list("ABC")
    at.run()
    assert not at.exception
    assert "b34_result" not in at.session_state
    assert len(calls) == 1 and len(exports) == 2 and not at.dataframe
    assert not at.get("plotly_chart") and not at.get("download_button")


@pytest.mark.parametrize("model", ["3PL", "4PL"])
def test_descriptive_table_and_csv_never_expose_research_p(model):
    table = U.result_table(result(model), model)
    assert table["Bootstrap p"].isna().all()
    exported = pd.read_csv(io.BytesIO(table.to_csv(index=False).encode("utf-8-sig")))
    assert exported["Bootstrap p"].isna().all()
    assert exported["LR (gözlenen)"].iloc[0] == 7.
    assert "kapalı" in exported.Durum.iloc[0]
    assert not U.inference_eligibility(model)[0]


def test_four_pl_screen_and_actual_csv_withhold_p_despite_study_flag(monkeypatch):
    monkeypatch.setattr(U, "_enabled", lambda: ({"3PL": True, "4PL": True}, {}))
    monkeypatch.setattr(U, "bootstrap_dif_test", lambda *a, **kw: result())
    exports = []
    monkeypatch.setattr(U.st, "download_button", lambda *a, **kw: exports.append(a[1]))
    at = AppTest.from_string('''
import streamlit as st
from ui.dif34_bootstrap_view import render
render(st.session_state.data, list("ABCD"), group_col="group")
''')
    at.session_state["data"] = frame()
    at.run(); at.radio(key="b34_model").set_value("4PL").run()
    at.button(key="b34_run").click().run()
    assert not at.exception and at.dataframe[0].value["Bootstrap p"].isna().all()
    assert pd.read_csv(io.BytesIO(exports[-1]))["Bootstrap p"].isna().all()
    assert at.get("plotly_chart")  # Descriptive bootstrap distribution survives.


def test_legacy_cached_tuple_is_discarded():
    at = AppTest.from_string('''
import streamlit as st
from ui.dif34_bootstrap_view import render
render(st.session_state.data, list("ABCD"), group_col="group")
''')
    at.session_state["data"] = frame()
    at.session_state["b34_result"] = (result(), "4PL")
    at.run()
    assert not at.exception and "b34_result" not in at.session_state and not at.dataframe
