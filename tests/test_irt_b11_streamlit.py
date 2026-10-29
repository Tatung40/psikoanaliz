"""3PL library route: AppTest isolation from the removed native B10/B11 controls."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

F = Path(__file__).parent / "fixtures" / "irt_b11"
APP = str(Path(__file__).resolve().parents[1] / "app.py")
ITEMS = [f"M{i}" for i in range(1, 13)]


def _small_dif():
    d = pd.read_csv(F / "data_dif3pl.csv")
    d = pd.concat([d[d.Grup == "R"].iloc[:700], d[d.Grup == "F"].iloc[:500]]).reset_index(drop=True)
    d.insert(0, "OGRENCI_NO", np.arange(len(d)))
    return d


def test_apptest_3pl_library_route_and_isolation():
    """The active library page must not expose the retired native B11 controls."""
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_file(APP, default_timeout=180)
    df = _small_dif()
    for key, value in dict(raw_df=df, scored_df=df.copy(), item_cols=ITEMS + ["Grup"],
                           id_col="OGRENCI_NO", group_col="Grup", is_dichotomous=True,
                           nav_page="MTK: 3PL").items(): at.session_state[key] = value
    at.run()
    assert not at.exception
    assert not [b for b in at.button if b.key and b.key.startswith(("b10_", "b11_"))]
    assert not [b for b in at.button if "MTK Analizini Başlat" in b.label]
    at.button(key="py_mirt_run_3PL").click().run()
    assert not at.exception
    entry = at.session_state["python_mirt_experimental_results"]["3PL"]
    assert entry["fit"].engine == "python_mirt" and entry["fit"].model == "3PL"
    assert entry["fit"].n_items == len(ITEMS)
    assert "3PL" not in at.session_state.get("irt_model_results", {})
    changed = df.copy(); changed.loc[0, ITEMS[0]] = 1 - changed.loc[0, ITEMS[0]]
    at.session_state["scored_df"] = changed
    at.run()
    assert not at.exception
    assert not any("Motor: python_mirt | Model: 3PL" in c.value for c in at.caption)
