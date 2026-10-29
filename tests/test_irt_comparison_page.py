"""'MTK: Karşılaştırma' compares Python mirt ML fits run on the 1PL–4PL pages (no native engine)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mirt")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _data():
    rng = np.random.default_rng(7)
    theta = rng.standard_normal(400)[:, None]
    a = np.linspace(.8, 1.8, 6); b = np.linspace(-1.2, 1.2, 6)
    p = 1 / (1 + np.exp(-a * (theta - b)))
    return pd.DataFrame((rng.random(p.shape) < p).astype(int), columns=[f"M{i}" for i in range(1, 7)])


def test_comparison_page_uses_library_fits_of_identical_data():
    df = _data()
    at = AppTest.from_file(APP, default_timeout=180)
    for key, value in dict(raw_df=df, scored_df=df.copy(), item_cols=list(df.columns), id_col=None,
                           group_col=None, is_dichotomous=True, nav_page="MTK: Karşılaştırma").items():
        at.session_state[key] = value
    at.run()
    assert not at.exception
    assert any("henüz bir MTK modeli çalıştırılmadı" in i.value for i in at.info)
    for model in ("1PL", "2PL"):
        at.session_state["nav_page"] = f"MTK: {model}"
        at.run()
        at.button(key=f"py_mirt_run_{model}").click().run()
        assert not at.exception
    at.session_state["nav_page"] = "MTK: Karşılaştırma"
    at.run()
    assert not at.exception
    fits = {m: at.session_state["python_mirt_experimental_results"][m]["fit"].fit for m in ("1PL", "2PL")}
    shown = at.dataframe[0].value.set_index("Model")
    for model, fit in fits.items():
        assert float(shown.loc[model, "AIC"]) == pytest.approx(round(fit.aic, 3), abs=1e-3)
        assert float(shown.loc[model, "BIC"]) == pytest.approx(round(fit.bic, 3), abs=1e-3)
    best = min(fits, key=lambda m: fits[m].aic)
    assert shown.loc["EN İYİ", "AIC"] == best
    changed = df.copy(); changed.iloc[0, 0] = 1 - changed.iloc[0, 0]
    at.session_state["scored_df"] = changed
    at.run()
    assert any("henüz bir MTK modeli çalıştırılmadı" in i.value for i in at.info)
