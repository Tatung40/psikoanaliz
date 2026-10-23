"""B10: helpers and AppTest interaction for the 1PL/2PL diagnostic tabs."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from ui import irt_b10_view as ui
from ui.irt_optimization import calibration_settings, run_irt_calibration

F = Path(__file__).parent / "fixtures" / "irt_b10"
APP = str(Path(__file__).resolve().parents[1] / "app.py")
ITEMS = [f"M{i}" for i in range(1, 13)]


def _dif_frame():
    d = pd.read_csv(F / "data_dif2pl_mixed.csv")
    d.insert(0, "OGRENCI_NO", np.arange(len(d)))
    return d


def _fit(frame, model, points=31):
    return run_irt_calibration(frame, calibration_settings(model, points, 200))


def test_result_key_changes_with_run_and_model():
    d = pd.read_csv(F / "data_irt2pl_ok.csv")
    r1, r2 = _fit(d, "2PL"), _fit(d, "2PL", points=21)
    assert ui.result_key("2PL", "fp", r1) != ui.result_key("2PL", "fp", r2)
    assert ui.result_key("1PL", "fp", r1) != ui.result_key("2PL", "fp", r1)


def test_global_fit_and_item_plot_use_shown_parameters():
    d = pd.read_csv(F / "data_irt2pl_ok.csv"); r = _fit(d, "2PL")
    gf = ui.run_global_fit(d, r, "2PL", "WLE", 4)
    assert gf["m2"]["available"] and gf["m2"]["free_parameters"] == 20 and gf["strata"]["Tabaka"].max() == 4
    rows = dict(zip(*ui.m2_rows(gf).T.values))
    assert rows["sd"] == 55 - 20
    fig = ui.item_fit_figure(gf, r, "M3")
    row = r["items"].set_index("Madde").loc["M3"]
    assert fig.data[0].y[80] == pytest.approx(1 / (1 + np.exp(-row["a"] * (0 - row["b"]))))   # grid[80] = 0


def test_q3_bootstrap_refits_with_the_shown_settings(monkeypatch):
    d = pd.read_csv(F / "data_irt1pl_ok.csv"); r = _fit(d, "1PL")
    seen = []
    original = ui.run_irt_calibration
    monkeypatch.setattr(ui, "run_irt_calibration", lambda data, s: seen.append(dict(s)) or original(data, s))
    q = ui.run_q3(d, r, "1PL", "EAP", 30, 2, seed=1)
    assert q["bootstrap"]["used"] + q["bootstrap"]["failed"] == 2 and len(seen) == 2
    assert all(s["model"] == "1PL" and s["points"] == 31 and s["max_iter"] == 200 for s in seen)
    assert q["ui_settings"]["person_method"] == "EAP"


def test_run_dif_rejects_group_as_item_and_builds_figures():
    d = _dif_frame()
    with pytest.raises(ValueError, match="madde olarak"):
        ui.run_dif(d, ITEMS + ["Grup"], "Grup", "R", "F", "1PL")
    out = ui.run_dif(d, ITEMS, "Grup", "R", "F", "1PL", purify=False)
    rows = dict(zip(*ui.dif_summary_rows(out).T.values))
    assert rows["N (R, referans)"] == 1200 and rows["Model"] == "1PL"
    fig = ui.dif_effect_figure(out)
    np.testing.assert_allclose(fig.data[0].x, out["table"]["İşaretli fark E_F[P_F − P_R]"])
    irf = ui.group_irf_figure(out, "M3")
    assert len(irf.data) == 2 and "M3" in irf.layout.title.text


def test_apptest_1pl_tabs_and_3pl_unchanged():
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_file(APP, default_timeout=300)
    df = _dif_frame()
    at.session_state["raw_df"] = df
    at.session_state["scored_df"] = df.copy()
    at.session_state["item_cols"] = ITEMS + ["Grup"]           # a mis-selected group column
    at.session_state["id_col"] = "OGRENCI_NO"
    at.session_state["group_col"] = "Grup"
    at.session_state["is_dichotomous"] = True
    at.session_state["nav_page"] = "MTK: 1PL"
    at.run()
    assert not at.exception
    next(b for b in at.button if "MTK Analizini Başlat" in b.label).click().run()
    assert not at.exception
    fit = at.session_state["irt_model_results"]["1PL"]["result"]
    assert list(fit["items"]["Madde"]) == ITEMS                 # group / ID never items
    at.button(key="b10_gf_run_1PL").click().run()
    assert not at.exception
    gf = at.session_state["irt_b10"][("gf", "1PL")]["result"]
    assert gf["m2"]["available"] and gf["m2"]["free_parameters"] == 12
    at.button(key="b10_q3_run_1PL").click().run()
    assert not at.exception and at.session_state["irt_b10"][("q3", "1PL")]["result"]["model"] == "1PL"
    assert at.selectbox(key="b10_dif_group_1PL").value == "Grup"
    at.selectbox(key="b10_dif_ref_1PL").select("R").run()
    at.selectbox(key="b10_dif_focal_1PL").select("F").run()
    at.button(key="b10_dif_run_1PL").click().run()
    assert not at.exception
    dif = at.session_state["irt_b10"][("dif", "1PL")]["result"]
    assert dif["model"] == "1PL" and "M3" in set(dif["table"].loc[dif["table"]["DIF işareti"], "Madde"])
    assert any("F (odak) − R (referans)" in m.value for m in at.markdown)
    at.checkbox(key="b10_dif_purify_1PL").uncheck().run()
    assert any("farklı grup veya ayarlarla" in i.value for i in at.info)
    at.checkbox(key="b10_dif_fixed_1PL").check().run()
    anchors = [c for c in ITEMS if c not in ("M3", "M7")]
    at.multiselect(key="b10_dif_anchors_1PL").set_value(anchors).run()
    assert at.checkbox(key="b10_dif_purify_1PL").disabled
    at.button(key="b10_dif_run_1PL").click().run()
    assert not at.exception
    fixed = at.session_state["irt_b10"][("dif", "1PL")]["result"]
    assert fixed["anchor_policy"] == "fixed" and fixed["anchors"] == anchors
    assert fixed["table"].set_index("Madde").loc[anchors, "p"].isna().all()
    at.multiselect(key="b10_dif_anchors_1PL").set_value(anchors[:-1]).run()
    assert any("farklı grup veya ayarlarla" in i.value for i in at.info)
    # a new calibration run invalidates the B10 results shown for the old one
    next(b for b in at.button if "MTK Analizini Başlat" in b.label).click().run()
    assert not any("F (odak) − R (referans)" in m.value for m in at.markdown)
    # 3PL page: no B10 tabs (B10 scope is 1PL/2PL only)
    assert ui.B10_MODELS == ("1PL", "2PL")
    at.session_state["nav_page"] = "MTK: 3PL"
    at.run()
    assert not at.exception
    assert not [b for b in at.button if b.key and b.key.startswith("b10_")]
