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
