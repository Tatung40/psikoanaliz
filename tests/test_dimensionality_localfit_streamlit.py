"""AppTest: iki yeni ekran çalışır, analiz eder ve indirme düğmeleri üretir."""
from pathlib import Path
import numpy as np
import pandas as pd
from streamlit.testing import v1 as st_testing

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _app(df, page):
    at = st_testing.AppTest.from_file(APP, default_timeout=180)
    at.session_state["raw_df"] = df
    at.session_state["scored_df"] = df.copy()
    at.session_state["item_cols"] = list(df.columns)
    at.session_state["id_col"] = None
    at.session_state["group_col"] = None
    at.session_state["is_dichotomous"] = True
    at.session_state["nav_page"] = page
    return at


def _data(n=500, k=8, seed=2):
    rng = np.random.default_rng(seed)
    th = rng.normal(size=n)
    p = 1 / (1 + np.exp(-1.5 * (th[:, None] - np.linspace(-1, 1, k))))
    return pd.DataFrame((rng.random(p.shape) < p).astype(float), columns=[f"M{i+1}" for i in range(k)])


def _run(page, run_key):
    at = _app(_data(), page)
    at.run(); assert not at.exception
    assert any(b.key in ("nav_dimensionality", "nav_local_fit") for b in at.button)
    next(b for b in at.button if b.key == run_key).click()
    if run_key == "dim_run":
        at.number_input(key="dim_reps").set_value(10)
    at.run(); assert not at.exception
    assert not at.error
    return at


def test_dimensionality_page():
    at = _run("Boyutluluk ve Faktör Analizi", "dim_run")
    assert any("Paralel analiz" in m.label for m in at.metric)


def test_local_fit_page():
    at = _run("Yerel Bağımsızlık ve Madde Uyumu", "lf_run")
    assert any("Ortalama Q3" in m.label for m in at.metric)


def test_local_fit_disabled_for_polytomous():
    df = pd.DataFrame(np.random.default_rng(0).integers(0, 4, (200, 5)).astype(float), columns=list("ABCDE"))
    at = _app(df, "Yerel Bağımsızlık ve Madde Uyumu"); at.run()
    assert not at.exception and any("S-X²" in e.value for e in at.error)
