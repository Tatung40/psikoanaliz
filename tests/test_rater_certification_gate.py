"""Regression: only independently validated outputs may be published."""
import pandas as pd
from core import rater_reliability as R
from ui import rater_reliability_view as UI


def test_unvalidated_primary_is_not_shown_or_exported(monkeypatch):
    data = R.from_wide(pd.DataFrame({"unit": ["a", "b", "c"], "r1": [0, 1, 1], "r2": [0, 0, 1]}), "unit", ["r1", "r2"])
    monkeypatch.setattr(R, "compute", lambda method, data: {"estimate": 0.5, "se": 0.1, "p": 0.02, "n": 3})
    monkeypatch.setattr(R, "validated_outputs", lambda method: {"estimate"})
    monkeypatch.setattr(R, "interpret", lambda *args: "Test")
    out = R.run("cohen_kappa", data)
    assert out["shown"] == {"estimate": 0.5, "n": 3}
    row = UI._result_row(out)
    assert row["SE"] == "—" and row["p"] == "—"
    # Details exports must derive from shown, not from the raw internal result.
    assert "se" in out["result"] and "se" not in out["shown"]


def test_missing_evidence_hides_all_statistics(monkeypatch):
    data = R.from_wide(pd.DataFrame({"unit": ["a", "b"], "r1": [0, 1], "r2": [1, 1]}), "unit", ["r1", "r2"])
    monkeypatch.setattr(R, "compute", lambda *args: {"estimate": 0.4, "p": 0.01})
    monkeypatch.setattr(R, "validated_outputs", lambda method: set())
    monkeypatch.setattr(R, "interpret", lambda *args: "Test")
    out = R.run("cohen_kappa", data)
    assert "estimate" not in out["shown"] and "p" not in out["shown"]


def test_narrative_png_and_matrix_follow_evidence(monkeypatch):
    import numpy as np
    from validation.rater_reliability.matrix import matrix, summary
    rows = matrix(); s = summary(rows)
    assert len({r["method"] for r in rows}) == 30 and s["certified"] == 0
    assert [(r["method"], r["output"]) for r in rows if r["status"] != "NUMERICALLY VALIDATED"] == [("icc_2_k", "ci95")]
    data = R.from_wide(pd.DataFrame({"u": list("abcd"), "x": [1., 2, 3, 4], "y": [1.5, 2, 2.5, 4.5]}), "u", ["x", "y"])
    monkeypatch.setattr(R, "validated_outputs", lambda m: set())
    out = R.run("bland_altman", data)
    assert "raporlanmamıştır" in out["interpretation"] and "bias" not in out["shown"]
    files = UI._exports({"bland_altman": out}, data, {"layout": "wide"})
    assert "bland_altman.png" not in files
    row = UI._result_row(dict(out, shown={"estimate": float("nan")}, method="cohen_kappa"))
    assert row["Değer"] == "—"


def test_duplicate_rater_columns_fail_closed():
    import pytest
    df = pd.DataFrame({"u": ["a", "b"], "r1": [1, 2]})
    with pytest.raises(R.RaterDataError, match="birden fazla"):
        R.from_wide(df, "u", ["r1", "r1"])
