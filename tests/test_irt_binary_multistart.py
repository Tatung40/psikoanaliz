"""Deterministic 3PL/4PL multi-start: selection rule, determinism, local-solution and replacement flags."""
import numpy as np
import pytest

pytest.importorskip("mirt")
from core import example_datasets as ex
from core.irt_binary_multistart import STARTS, multistart, objective
from core.irt_python_mirt_adapter import fit_python_mirt


def _x(case):
    b = ex.materialize(case)
    return b["data"][b["items"]].to_numpy(int)


def test_4pl_ml_local_solution_is_detected_and_best_objective_selected():
    x = _x("irt_013")
    best, table, notes = multistart(x, "4PL", "ML")
    assert len(table) == 2 + len(STARTS) and table["Seçildi"].sum() == 1
    conv = table[table["Yakınsadı"]]
    assert objective(best.fit, "ML") == pytest.approx(conv["Amaç"].max(), abs=1e-6)
    assert (conv["En iyiden fark"] > .5).any() and any("ayrı çözümde durdu" in n for n in notes)
    again = multistart(x, "4PL", "ML")[1]
    np.testing.assert_array_equal(table["Amaç"].to_numpy(), again["Amaç"].to_numpy())


def test_default_replaced_only_when_clearly_worse_and_map_is_start_insensitive():
    x = _x("irt_014")
    best, table, notes = multistart(x, "4PL", "ML")
    default = fit_python_mirt(x, model="4PL", estimation="ML").fit
    gap = objective(best.fit, "ML") - objective(default, "ML")
    assert gap > 1e-3 and any("Varsayılan başlangıç" in n for n in notes)
    _, table_map, _ = multistart(x, "4PL", "MAP")
    assert table_map[table_map["Yakınsadı"]]["En iyiden fark"].max() < 1e-2
    assert table_map[table_map["Yakınsadı"]]["En büyük IRF farkı"].max() < .02


def test_guards():
    with pytest.raises(ValueError):
        multistart(_x("irt_014"), "2PL")
    with pytest.raises(ValueError):
        fit_python_mirt(_x("irt_014")[:, :5], model="2PL", accelerate="squarem")


def test_screen_multistart_is_opt_in_and_shows_table_and_selected_fit():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    b = ex.materialize("irt_014"); x = b["data"][b["items"]].astype(int)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=400)
    for key, value in dict(raw_df=x, scored_df=x.copy(), item_cols=list(x.columns), id_col=None, group_col=None,
                           is_dichotomous=True, nav_page="MTK: Python mirt 4PL").items():
        at.session_state[key] = value
    at.run()
    at.radio(key="python_mirt_estimation_4PL").set_value("ML").run()
    at.button(key="python_mirt_start_4PL").click().run()
    assert not at.exception and at.session_state["python_mirt_fit_4PL"]["multistart"] is None
    at.checkbox(key="python_mirt_multistart_4PL").check().run()
    at.button(key="python_mirt_start_4PL").click().run()
    assert not at.exception
    table, notes = at.session_state["python_mirt_fit_4PL"]["multistart"]
    assert any("Çoklu başlangıç denetimi" in m.value for m in at.markdown)
    assert any("Varsayılan başlangıç" in w.value for w in at.warning)
    chosen = at.session_state["python_mirt_fit_4PL"]["outcome"].fit
    assert objective(chosen, "ML") == pytest.approx(table.loc[table["Seçildi"], "Amaç"].item(), abs=1e-9)
