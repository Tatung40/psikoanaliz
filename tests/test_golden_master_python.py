"""Unit checks of the golden-master comparator used by the same-run CI equivalence gate.

The gate itself (.github/workflows/pytest.yml, "Golden-master equivalence") captures
tools/golden_master.py sections from the pre-removal main and from the current tree on
the same runner and requires zero differences.
"""
import math

import numpy as np
import pandas as pd

from tools import golden_master as gm


def test_encode_drops_volatile_columns_and_key_value_rows():
    frame = pd.DataFrame({"Setting": ["Model", "Python", "mirt_version"], "Value": ["2PL", "3.11", "1.2.0"]})
    assert gm.encode(frame)["rows"] == [["Model", "2PL"]]
    assert gm.encode(pd.DataFrame({"a": [1.5], "Python": ["3.13"]}))["__frame__"] == ["a"]
    assert gm.encode({"x": np.float64(np.nan), "versions": {"numpy": "2"}}) == {"x": {"__float__": "nan"}}


def test_compare_is_exact_by_default_and_reports_every_kind_of_difference():
    base = gm.encode({"t": pd.DataFrame({"M": ["A", "B"], "b": [0.1, -0.2]}), "n": 3})
    assert gm.compare(base, base, rtol=0, atol=0) == []
    moved = gm.encode({"t": pd.DataFrame({"M": ["A", "B"], "b": [0.1, -0.2 + 1e-15]}), "n": 3})
    assert len(gm.compare(base, moved, rtol=0, atol=0)) == 1
    assert gm.compare(base, moved, rtol=1e-12, atol=0) == []
    reordered = gm.encode({"t": pd.DataFrame({"M": ["B", "A"], "b": [-0.2, 0.1]}), "n": 3})
    assert gm.compare(base, reordered, rtol=0, atol=0)
    assert gm.compare(base, gm.encode({"t": base["t"]}), rtol=0, atol=0)
    assert not math.isnan(gm.encode(1.25))


def test_every_section_is_callable_and_capture_round_trips(tmp_path):
    assert set(gm.SECTIONS) >= {"python_mirt_models", "generalizability", "example_library", "b14", "python_dif"}
    data = {"ctt": gm.run_section("ctt")}
    gm.save(data, tmp_path / "m.json.gz")
    assert gm.compare(data, gm.load(tmp_path / "m.json.gz"), rtol=0, atol=0) == []
