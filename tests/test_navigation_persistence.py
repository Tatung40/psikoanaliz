"""Results stay on screen; pages survive reruns and server-session resets (URL), no silent home jumps."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _data(poly=False):
    rng = np.random.default_rng(3); th = rng.normal(size=300)
    if poly:
        x = (rng.random((300, 6))[..., None] < 1 / (1 + np.exp(-(th[:, None, None] - np.array([-1., 0, 1]))))).sum(2)
    else:
        x = (rng.random((300, 6)) < 1 / (1 + np.exp(-(th[:, None] - np.linspace(-1, 1, 6))))).astype(int)
    return pd.DataFrame(x, columns=[f"M{i}" for i in range(6)])


def _app(page, df=None):
    at = st_testing.AppTest.from_file(APP, default_timeout=240)
    if df is not None:
        for k, v in (("raw_df", df), ("scored_df", df.copy()), ("item_cols", list(df.columns)), ("id_col", None),
                     ("group_col", None), ("is_dichotomous", df.max().max() <= 1)):
            at.session_state[k] = v
    at.session_state["nav_page"] = page
    return at.run()


def test_1pl_results_persist_across_reruns_without_home_redirect():
    pytest.importorskip("mirt")
    at = _app("MTK: 1PL", _data())
    at.button(key="py_mirt_run_1PL").click().run()
    assert not at.exception and at.session_state["python_mirt_experimental_results"]
    for _ in range(2):
        at.run()
        assert at.session_state["nav_page"] == "MTK: 1PL" and at.query_params["sayfa"] in ("MTK: 1PL", ["MTK: 1PL"])
        assert at.session_state["python_mirt_experimental_results"]


def test_new_session_restores_page_from_url_and_warns():
    at = st_testing.AppTest.from_file(APP, default_timeout=120)
    at.query_params["sayfa"] = "MTK: 1PL"
    at.run()
    assert not at.exception and at.session_state["nav_page"] == "MTK: 1PL"
    assert any("Sunucu oturumu yenilendi" in w.value for w in at.warning)


def test_manual_navigation_clear_data_and_unknown_page():
    at = _app("4. Test İstatistikleri", _data())
    assert not at.exception and at.query_params["sayfa"] in ("4. Test İstatistikleri", ["4. Test İstatistikleri"])
    at.button(key="clear_privacy_session").click().run()
    assert at.session_state["nav_page"] == "4. Test İstatistikleri" and at.session_state["raw_df"] is None
    at.button(key="nav_irt_b14").click().run()
    assert at.session_state["nav_page"] == "MTK: Çok Kategorili B14"
    bad = st_testing.AppTest.from_file(APP, default_timeout=120)
    bad.query_params["sayfa"] = "yok-boyle-sayfa"
    bad.run()
    assert not bad.exception and bad.session_state["nav_page"] == "1. Ana Sayfa"


@pytest.mark.parametrize("page,poly", [("4. Test İstatistikleri", False), ("MTK: Çok Kategorili B14", True)])
def test_ctt_and_polytomous_pages_stay_put(page, poly):
    at = _app(page, _data(poly))
    at.run()
    assert not at.exception and at.session_state["nav_page"] == page
