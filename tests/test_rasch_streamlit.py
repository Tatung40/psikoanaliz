"""B8: Streamlit Rasch page helpers and a real AppTest interaction."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from ui import rasch_view as ui

FIX = Path(__file__).parent / "fixtures" / "rasch"
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _data():
    return pd.read_csv(FIX / "data_rasch_missing.csv", na_values="NA")


def test_screen_items_allows_missing_but_rejects_bad_columns():
    df = _data().copy()
    df["SABIT"] = 1; df["KOD"] = "a"; df.loc[0, "M2"] = 3
    usable, report = ui.screen_items(df, list(df.columns))
    assert "M1" in usable and "SABIT" not in usable and "KOD" not in usable and "M2" not in usable
    status = report.set_index("Madde")["Durum"]
    assert status["SABIT"] == "tüm yanıtlar aynı" and status["KOD"] == "0/1 dışı değer" and status["M2"] == "0/1 dışı değer"
    assert report.set_index("Madde").loc["M1", "Eksik"] == int(df["M1"].isna().sum())


def test_run_rasch_records_settings_and_outputs():
    df = _data()
    s = ui.rasch_settings(points=31, mnsq_label="0,5–1,5 (ölçme için üretken aralık)")
    r = ui.run_rasch(df, list(df.columns), s)
    assert r["ui_settings"]["points"] == 31 and r["fit"]["points"] == 31
    assert r["settings"]["mnsq_range"] == (0.5, 1.5) and r["ui_settings"]["seconds"] >= 0
    table = dict(zip(*ui.summary_rows(r).T.values))
    assert table["Yakınsama"] == "Evet" and table["Gauss–Hermite düğümü"] == 31
    assert set(ui.misfit_persons(r)["Uyum"]) <= {"İnceleme önerilir"}


def test_wright_map_uses_final_estimates():
    df = _data()
    r = ui.run_rasch(df, list(df.columns), ui.rasch_settings())
    fig = ui.wright_map(r)
    bars, items = fig.data
    assert int(np.sum(bars.customdata)) == len(r["persons"])
    np.testing.assert_allclose(items.y, r["fit"]["b"])
    assert list(items.text) == list(r["items"]["Madde"])


def test_fingerprint_changes_with_data():
    df = _data(); cols = list(df.columns)
    other = df.copy(); other.iloc[0, 0] = 1 - (other.iloc[0, 0] if pd.notna(other.iloc[0, 0]) else 0)
    assert ui.fingerprint(df, cols) != ui.fingerprint(other, cols)
    assert ui.fingerprint(df, cols) != ui.fingerprint(df, cols[:-1])


def test_apptest_rasch_page_runs_and_shows_results():
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_file(APP, default_timeout=120)
    df = _data()
    at.session_state["raw_df"] = df
    at.session_state["scored_df"] = df.copy()
    at.session_state["item_cols"] = list(df.columns)
    at.session_state["nav_page"] = "MTK: Rasch"
    at.run()
    assert not at.exception
    assert any("MAR" in i.value for i in at.info)                         # missing-data notice
    next(b for b in at.button if "Rasch Analizini Başlat" in b.label).click().run()
    assert not at.exception
    assert any("Rasch analizi tamamlandı" in s.value for s in at.success)
    stored = at.session_state["rasch_result"]["result"]
    assert stored["fit"]["converged"] and stored["ui_settings"]["points"] == 41
    assert any(m.label == "Örtük SS (σ)" for m in at.metric)
    assert any("Infit MNSQ" in df_el.value.columns for df_el in at.dataframe)
    at.selectbox(key="rasch_mnsq").select("0,5–1,5 (ölçme için üretken aralık)").run()
    assert any("farklı ayarlarla üretildi" in i.value for i in at.info)
