import pytest
from core.academic_result_schema import build_result
from core.academic_visual_catalog import build_visual_catalog


@pytest.mark.parametrize("kind,table,x,y", [
    ("item_calibration", "category_curves", "Theta", "Olasılık"),
    ("DIF", "Difference", "Theta", "Focal_minus_reference"),
    ("G_and_D_study", "variance", "component", "Variance"),
])
def test_catalog_tracks_existing_numeric_data(kind, table, x, y):
    result = build_result(family="GT" if kind == "G_and_D_study" else "IRT",
        model="p x i" if kind == "G_and_D_study" else "2PL", n=30,
        tables={table: [{x: 0, y: 0.8}]}, metadata={"analysis_type": kind})
    manifest = build_visual_catalog(result)
    assert manifest["tables"][0]["row_count"] == 1
    assert manifest["figures"][0]["source_table"] == table
    assert manifest["figures"][0]["render_status"] == "NOT_RENDERED"
    assert manifest["figures"][0]["rendered_file"] is None


def test_missing_plot_input_is_not_fabricated():
    result = build_result(family="IRT", model="RSM", n=40,
        tables={"category_curves": [{"Theta": 0.0}]},
        metadata={"analysis_type": "item_calibration"})
    manifest = build_visual_catalog(result)
    assert not manifest["figures"]
    assert manifest["warnings"]
