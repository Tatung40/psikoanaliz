"""B9: Streamlit helpers and AppTest interaction for model fit, Q3 and DIF tabs."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from ui import rasch_view as ui

FIX = Path(__file__).parent / "fixtures" / "rasch_b9"
APP = str(Path(__file__).resolve().parents[1] / "app.py")
ITEMS = [f"M{i}" for i in range(1, 13)]


def _dif_data():
    df = pd.read_csv(FIX / "data_dif_known.csv", na_values="NA")
    df.insert(0, "OGRENCI_NO", np.arange(len(df)))
    return df


def test_group_candidates_exclude_items_and_high_cardinality_columns():
    df = _dif_data(); df["Not"] = [f"serbest metin {i}" for i in range(len(df))]
    cands = ui.group_candidates(df.drop(columns=["OGRENCI_NO"]), ITEMS)
    assert cands == ["Grup"]                                   # items, ID and free text are not groups


def test_run_dif_rejects_group_as_item_and_records_settings():
    df = _dif_data()
    with pytest.raises(ValueError, match="madde olarak"):
        ui.run_dif(df, ITEMS + ["Grup"], "Grup", "R", "F")
    out = ui.run_dif(df, ITEMS, "Grup", "R", "F", alpha=.05, purify=True, max_rounds=3)
    assert out["ui_settings"]["reference"] == "R" and out["ui_settings"]["max_rounds"] == 3
    assert set(out["table"].loc[out["table"]["DIF işareti"], "Madde"]) == {"M3", "M8"}
    rows = dict(zip(*ui.dif_summary_rows(out).T.values))
    assert rows["N (R, referans)"] == 900 and rows["N (F, odak)"] == 500
    fig = ui.dif_figure(out)
    np.testing.assert_allclose(fig.data[0].x, out["table"]["Δ (b_odak − b_ref)"])
    assert "F − R" in fig.layout.title.text


def test_run_q3_and_global_fit_use_the_b8_estimates():
    df = pd.read_csv(FIX / "data_ld_pair.csv", na_values="NA")
    r = ui.run_rasch(df, list(df.columns), ui.rasch_settings())
    q = ui.run_q3(df, list(df.columns), r, "WLE", 30, 0)
    assert tuple(q["pairs"].iloc[0][["Madde 1", "Madde 2"]]) == ("M2", "M3")
    np.testing.assert_allclose(ui.q3_heatmap(q).data[0].z, q["adjusted"].to_numpy())
    gf = ui.run_global_fit(r, strata=4)
    assert gf["m2"]["available"] and gf["ui_settings"]["strata"] == 4
    assert gf["strata"]["Tabaka"].max() == 4
    m2 = dict(zip(*ui.m2_rows(gf).T.values))
    assert m2["sd"] == 10 + 45 - 11 and m2["Serbest parametre (b + σ)"] == 11
    assert int(ui.raw_score_figure(gf).data[0].y.sum()) == gf["raw_complete_n"] == len(df)


def test_m2_rows_reports_reason_when_unavailable():
    rows = ui.m2_rows({"m2": {"available": False, "reason": "eksiksiz veri gerektirir"}})
    assert "eksiksiz" in rows["Değer"].iloc[0]


def test_apptest_b9_tabs_fit_q3_and_dif():
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_file(APP, default_timeout=300)
    df = _dif_data()
    at.session_state["raw_df"] = df
    at.session_state["scored_df"] = df.copy()
    at.session_state["item_cols"] = ITEMS + ["Grup"]           # a mis-selected group column
    at.session_state["id_col"] = "OGRENCI_NO"
    at.session_state["group_col"] = "Grup"
    at.session_state["nav_page"] = "MTK: Rasch"
    at.run()
    assert not at.exception
    next(b for b in at.button if "Rasch Analizini Başlat" in b.label).click().run()
    assert not at.exception
    b8 = at.session_state["rasch_result"]["result"]
    assert list(b8["items"]["Madde"]) == ITEMS                 # Grup / ID never treated as items
    at.button(key="rasch_gf_run").click().run()
    assert not at.exception
    gf = at.session_state["rasch_gf"]["result"]
    assert not gf["m2"]["available"] and "eksiksiz" in gf["m2"]["reason"]   # 8% MCAR: M2 withheld
    at.button(key="rasch_q3_run").click().run()
    assert not at.exception
    assert at.session_state["rasch_q3"]["result"]["ui_settings"]["bootstrap"] == 0
    assert at.selectbox(key="rasch_dif_group").value == "Grup"
    at.selectbox(key="rasch_dif_ref").select("R").run()
    at.selectbox(key="rasch_dif_focal").select("F").run()
    at.button(key="rasch_dif_run").click().run()
    assert not at.exception
    dif = at.session_state["rasch_dif"]["result"]
    assert dif["reference"] == "R" and dif["focal"] == "F"
    assert set(dif["table"].loc[dif["table"]["DIF işareti"], "Madde"]) == {"M3", "M8"}
    assert any("F (odak) − R (referans)" in m.value for m in at.markdown)
    at.checkbox(key="rasch_dif_purify").uncheck().run()
    assert any("farklı grup veya ayarlarla" in i.value for i in at.info)
    # A new B8 run with other settings invalidates the B9 results (session isolation by key).
    at.select_slider(key="rasch_points").set_value(31).run()
    next(b for b in at.button if "Rasch Analizini Başlat" in b.label).click().run()
    assert not any("F (odak) − R (referans)" in m.value for m in at.markdown)
