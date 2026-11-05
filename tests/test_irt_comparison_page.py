"""'MTK: Karşılaştırma' fits 1PL–4PL by ML in one click on identical data (no native engine)."""
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


def test_comparison_page_fits_all_four_models_on_identical_data():
    df = _data()
    at = AppTest.from_file(APP, default_timeout=600)
    for key, value in dict(raw_df=df, scored_df=df.copy(), item_cols=list(df.columns), id_col=None,
                           group_col=None, is_dichotomous=True, nav_page="MTK: Karşılaştırma").items():
        at.session_state[key] = value
    at.run()
    assert not at.exception
    assert any("düğmeye basın" in i.value for i in at.info)
    at.button(key="irt_compare_run").click().run()
    assert not at.exception
    rows = {r["Model"]: r for r in at.session_state["irt_model_comparison"]["rows"]}
    shown = at.dataframe[0].value.set_index("Model")
    assert list(shown.index) == ["1PL", "2PL", "3PL", "4PL"]
    for model, row in rows.items():
        if row["AIC"] is not None:
            assert float(shown.loc[model, "AIC"]) == pytest.approx(row["AIC"], abs=1e-3)
            assert float(shown.loc[model, "BIC"]) == pytest.approx(row["BIC"], abs=1e-3)
    best = min((r for r in rows.values() if r["eligible"]), key=lambda r: r["BIC"])["Model"]
    assert shown.loc[best, "ΔBIC"] == "0.000"
    changed = df.copy(); changed.iloc[0, 0] = 1 - changed.iloc[0, 0]
    at.session_state["scored_df"] = changed
    at.run()
    assert any("geçersizleştirildi" in i.value for i in at.info) and not at.dataframe
