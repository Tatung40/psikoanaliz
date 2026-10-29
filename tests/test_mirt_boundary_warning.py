"""MIRT screen states a box-bound estimate explicitly instead of only listing it in a table."""
from pathlib import Path

import pytest

pytest.importorskip("mirt")
from streamlit.testing.v1 import AppTest
from core import example_datasets as ex


def test_bifactor_bound_estimate_is_warned_on_screen():
    # Teaching bifactor 'Temel' (N=300, k=12): converges at screen defaults with one loading on the box bound.
    bundle = ex.materialize("mirt_051")
    data = bundle["data"]
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=240)
    for key, value in dict(raw_df=data, scored_df=None, item_cols=bundle["items"], id_col=bundle["id"],
                           nav_page="Çok Boyutlu MTK (MIRT)").items():
        at.session_state[key] = value
    at.run()
    at.selectbox(key="md_model").set_value("Bifactor").run()
    at.selectbox(key="md_dims").set_value(4).run()
    at.button(key="md_run").click().run()
    assert not at.exception
    import ui.irt_multidim_view as view
    result = at.session_state[view.KEY]["result"]
    assert result["metadata"]["Converged"] and result["metadata"]["At_box_bound"]
    assert any("Kutu sınırında parametre var" in w.value for w in at.warning)
