"""B14 optional results (model comparison, diagnostics) are downloadable, packaged and tied to one fit."""
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("mirt")
from streamlit.testing.v1 import AppTest
from core import example_datasets as ex
from reports.irt_b14_report import export_zip, report_tables


def _grm():
    spec = next(d for d in ex.registry().values() if d.family == "irt" and d.model == "GRM" and d.level == 0)
    bundle = ex.materialize(spec.id)
    return bundle["data"][bundle["items"]]


def test_comparison_and_diagnostics_reach_screen_exports_and_are_dropped_on_data_change():
    data = _grm()
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=300)
    for key, value in {"raw_df": data, "scored_df": None, "item_cols": list(data.columns), "id_col": None,
                       "group_col": None, "nav_page": "MTK: Çok Kategorili B14"}.items():
        at.session_state[key] = value
    at.run()
    at.radio(key="b14_model").set_value("GRM").run()
    at.button(key="b14_run").click().run()
    at.button(key="b14_compare_run").click().run()
    at.button(key="b14_diagnostics_run").click().run()
    assert not at.exception
    current = at.session_state["b14_current"]
    token, (table, lr) = at.session_state["b14_compare"]
    assert token == current["token"] and at.session_state["b14_diagnostics"]["token"] == token
    assert any(b.key == "b14_compare_csv" for b in at.get("download_button"))
    assert set(table["Model"]) == {"GRM", "PCM", "GPCM"} and lr["Karşılaştırma"].startswith("PCM")
    diagnostic = at.session_state["b14_diagnostics"]
    extras = {"Tanı Durumu": diagnostic["status"], **{"Tanı - " + k: v for k, v in diagnostic["tables"].items()},
              "Model Karşılaştırması": table}
    frames = report_tables(current["result"], extras)
    assert frames["Model Karşılaştırması"]["AIC"].tolist() == table["AIC"].tolist()
    names = zipfile.ZipFile(io.BytesIO(export_zip(current["result"], extras))).namelist()
    assert {"model_karsilastirma.csv", "tani_durumu.csv", "B14_rapor.xlsx"} <= set(names)
    sheets = pd.read_excel(io.BytesIO(zipfile.ZipFile(io.BytesIO(export_zip(current["result"], extras))).read("B14_rapor.xlsx")),
                           sheet_name=None)
    assert "Model Karşılaştırması" in sheets and any(name.startswith("Tanı - ") for name in sheets)
    at.button(key="b14_zip_make").click().run()
    assert any(b.key == "b14_zip_dl" for b in at.get("download_button"))
    changed = data.copy()
    changed.iloc[0, 0] = (changed.iloc[0, 0] + 1) % 4
    at.session_state["raw_df"] = changed
    at.button(key="b14_run").click().run()
    assert not at.exception
    keys = {b.key for b in at.get("download_button")}
    assert "b14_compare_csv" not in keys and "b14_zip_dl" not in keys
    assert not any(df.value.equals(table) for df in at.dataframe)
