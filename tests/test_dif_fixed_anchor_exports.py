import io
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from ui.irt_b10_view import dif_export_table, dif_effect_figure


def test_dif_exports_escape_formulas_preserve_numbers_and_exclude_respondents():
    source = pd.DataFrame({"Madde": ["=1+1", " +external", "\t@unsafe"],
                           "Test çapaları": ["=label, M2", "M1, M2", "M1, M2"],
                           "p": [.012, np.nan, .5]})
    out = dif_export_table({"table": source, "respondents": ["private_identifier"]})
    assert out.Madde.tolist() == ["'=1+1", "' +external", "'\t@unsafe"]
    assert source.loc[0, "Madde"] == "=1+1"  # charts and estimates retain real labels
    np.testing.assert_allclose(out.p, source.p, equal_nan=True)
    assert "private_identifier" not in out.to_csv(index=False)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        out.to_excel(writer, index=False)
    sheet = load_workbook(io.BytesIO(buffer.getvalue())).active
    assert sheet["A2"].data_type == "s" and sheet["B2"].data_type == "s"
    assert sheet["C2"].value == .012


def test_inconclusive_item_table_has_a_safe_empty_effect_plot():
    figure = dif_effect_figure({"table": pd.DataFrame({"Madde": ["M1"], "DIF işareti": [False]}),
                                "model": "2PL", "reference": "R", "focal": "F"})
    assert len(figure.data[0].x) == 0
