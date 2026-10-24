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
