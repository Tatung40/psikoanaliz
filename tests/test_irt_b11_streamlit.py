"""B11: helpers and AppTest interaction for the 3PL/4PL diagnostic tabs."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from ui import irt_b11_view as ui
from ui.irt_optimization import calibration_settings, run_irt_calibration

F = Path(__file__).parent / "fixtures" / "irt_b11"
APP = str(Path(__file__).resolve().parents[1] / "app.py")
ITEMS = [f"M{i}" for i in range(1, 13)]


def _fit(frame, model, points=21, max_iter=150):
    return run_irt_calibration(frame, calibration_settings(model, points, max_iter))


def _small_dif():
    d = pd.read_csv(F / "data_dif3pl.csv")
    d = pd.concat([d[d.Grup == "R"].iloc[:700], d[d.Grup == "F"].iloc[:500]]).reset_index(drop=True)
    d.insert(0, "OGRENCI_NO", np.arange(len(d)))
    return d


def test_global_fit_uses_full_irf_and_reports_boundary_policy():
    d = pd.read_csv(F / "data_irt3pl_easy.csv"); r = _fit(d, "3PL")
    gf = ui.run_global_fit(d, r, "3PL", "EAP", 4)
    m = gf["m2"]
    assert m["available"] and m["free_parameters"] + m["fixed_parameters"] == 36
    rows = dict(zip(*ui.m2_rows(gf).T.values))
    if m["conditional"]:                                   # bound-active parameters: no p-value
        assert "sınır" in str(rows["p (asimptotik χ²)"]) and not gf["boundary"].empty
    fig = ui.item_fit_figure(gf, r, "M2")
    row = r["items"].set_index("Madde").loc["M2"]
    assert fig.data[0].y[0] == pytest.approx(row["c"] + (1 - row["c"]) / (1 + np.exp(-row["a"] * (-4 - row["b"]))))


def test_q3_bootstrap_refits_with_the_shown_settings(monkeypatch):
    d = pd.read_csv(F / "data_irt3pl_c0.csv"); r = _fit(d, "3PL")
    seen = []
    original = ui.run_irt_calibration
    monkeypatch.setattr(ui, "run_irt_calibration", lambda data, s: seen.append(dict(s)) or original(data, s))
    q = ui.run_q3(d, r, "3PL", "WLE", 30, 2, seed=3)
    assert q["bootstrap"]["used"] + q["bootstrap"]["failed"] == 2 and len(seen) == 2
    assert all(s["model"] == "3PL" and s["points"] == 21 and s["max_iter"] == 150 for s in seen)
    assert q["model"] == "3PL" and q["ui_settings"]["person_method"] == "WLE"


def test_4pl_dif_is_descriptive_only():
    assert ui.DIF_INFERENTIAL == {"3PL": True, "4PL": False}
    d = pd.read_csv(F / "data_dif4pl.csv")
    d = pd.concat([d[d.Grup == "R"].iloc[:600], d[d.Grup == "F"].iloc[:400]]).reset_index(drop=True)
    r = {"a_upper": 4., "d_lower": .65}
    out = ui.run_dif(d, ITEMS[:8], "Grup", "R", "F", "4PL", r, points=15)
    assert not out["inferential"] and out["table"]["p"].isna().all() and not out["table"]["DIF işareti"].any()
    assert set(out["table"]["Çıkarım"]) <= {"Betimsel", "Test edilemedi"} and not out["purify"]
    rows = dict(zip(*ui.dif_summary_rows(out).T.values))
    assert rows["Çıkarım"] == "Yalnızca betimsel"
    with pytest.raises(ValueError, match="madde olarak"):
        ui.run_dif(d, ITEMS + ["Grup"], "Grup", "R", "F", "4PL", r)


def test_apptest_3pl_tabs_and_isolation():
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_file(APP, default_timeout=600)
    df = _small_dif()
    at.session_state["raw_df"] = df
    at.session_state["scored_df"] = df.copy()
    at.session_state["item_cols"] = ITEMS + ["Grup"]
    at.session_state["id_col"] = "OGRENCI_NO"
    at.session_state["group_col"] = "Grup"
    at.session_state["is_dichotomous"] = True
    at.session_state["nav_page"] = "MTK: 3PL"
    at.run()
    assert not at.exception
    next(b for b in at.button if "MTK Analizini Başlat" in b.label).click().run()
    assert not at.exception
    fit = at.session_state["irt_model_results"]["3PL"]["result"]
    assert list(fit["items"]["Madde"]) == ITEMS
    at.button(key="b11_gf_run_3PL").click().run()
    assert not at.exception and at.session_state["irt_b11"][("gf", "3PL")]["result"]["model"] == "3PL"
    at.button(key="b11_q3_run_3PL").click().run()
    assert not at.exception and at.session_state["irt_b11"][("q3", "3PL")]["result"]["model"] == "3PL"
    at.selectbox(key="b11_dif_ref_3PL").select("R").run()
    at.selectbox(key="b11_dif_focal_3PL").select("F").run()
    at.select_slider(key="b11_dif_rounds_3PL").set_value(1).run()
    at.button(key="b11_dif_run_3PL").click().run()
    assert not at.exception
    dif = at.session_state["irt_b11"][("dif", "3PL")]["result"]
    assert dif["model"] == "3PL" and dif["inferential"]
    assert any("F (odak) − R (referans)" in m.value for m in at.markdown)
    at.checkbox(key="b11_dif_purify_3PL").uncheck().run()
    assert any("farklı grup veya ayarlarla" in i.value for i in at.info)
    next(b for b in at.button if "MTK Analizini Başlat" in b.label).click().run()      # new run hides old results
    assert not any("F (odak) − R (referans)" in m.value for m in at.markdown)
    assert not [b for b in at.button if b.key and b.key.startswith("b10_")]          # 1PL/2PL tabs not on 3PL page
