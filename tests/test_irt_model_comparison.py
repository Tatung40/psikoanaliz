"""One-click 1PL–4PL comparison: table, AIC/BIC arithmetic, eligibility, interpretation, cache."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.irt_model_comparison import MODELS, SOLVER, comparison_table, interpretation, summarize_fit

ROOT = Path(__file__).resolve().parents[1]


def _fake(ll, k, n=500, converged=True):
    return SimpleNamespace(log_likelihood=ll, n_parameters=k, n_observations=n, converged=converged,
                           aic=-2 * ll + 2 * k, bic=-2 * ll + k * math.log(n), coef=lambda: pd.DataFrame())


def _rows(lls, conv=(True, True, True, True), n=500):
    return [summarize_fit(m, _fake(ll, k, n, c), n) for m, ll, k, c in zip(MODELS, lls, (10, 20, 30, 40), conv)]


def test_real_lsat7_fits_order_counts_and_information_criteria():
    pytest.importorskip("mirt")
    from core.irt_python_mirt_adapter import fit_python_mirt
    ref = json.loads((ROOT / "tests/fixtures/psychometrics_reference/reference_values.json").read_text())["lsat7"]
    x = np.array([[int(v) for v in key.split(",")] for key, n in ref["patterns"].items() for _ in range(n)])
    rows = [summarize_fit(m, fit_python_mirt(x, model=m, estimation="ML", **SOLVER).fit, len(x)) for m in MODELS]
    table = comparison_table(rows)
    assert list(table["Model"]) == ["1PL", "2PL", "3PL", "4PL"] and len(table) == 4
    assert [r["k"] for r in rows[:3]] == [5, 10, 15]
    for r in rows:
        if r["k"] is not None:
            assert abs(r["AIC"] - (-2 * r["LogLik"] + 2 * r["k"])) < 1e-9
            assert abs(r["BIC"] - (-2 * r["LogLik"] + r["k"] * math.log(1000))) < 1e-9
    eligible = [r for r in rows if r["eligible"]]
    assert eligible and min(float(v) for v in table.loc[table["ΔBIC"] != "—", "ΔBIC"]) == 0
    assert not rows[3]["eligible"]                      # 4PL ML never selected


def test_delta_values_and_bic_selection_from_eligible_models():
    rows = _rows([-3000., -2950., -2948., -2900.], conv=(True, True, True, False))
    table = comparison_table(rows).set_index("Model")
    aic = {m: -2 * ll + 2 * k for m, ll, k in zip(MODELS, (-3000, -2950, -2948), (10, 20, 30))}
    bic = {m: -2 * ll + k * math.log(500) for m, ll, k in zip(MODELS, (-3000, -2950, -2948), (10, 20, 30))}
    for m in ("1PL", "2PL", "3PL"):
        assert table.loc[m, "ΔAIC"] == f"{aic[m] - min(aic.values()):.3f}"
        assert table.loc[m, "ΔBIC"] == f"{bic[m] - min(bic.values()):.3f}"
    assert table.loc["4PL", "ΔBIC"] == "—" and table.loc["4PL", "Yakınsama"] == "Hayır"
    text = interpretation(rows)
    assert "en küçük BIC değeri 2PL" in text and "4PL (yakınsamadı)" in text
    assert text.count("\n") == 0


def test_failed_or_unidentified_4pl_is_never_recommended_even_with_smallest_bic():
    rows = _rows([-3000., -2990., -2985., -2000.])      # converged 4PL with absurdly better LL
    assert not rows[3]["eligible"]
    assert "en küçük BIC değeri 4PL" not in interpretation(rows)
    failed = _rows([-3000., -2990., -2985., -2000.])[:3] + [summarize_fit("4PL", None, 500)]
    assert comparison_table(failed).iloc[3].tolist()[1:5] == ["—"] * 4


def test_incompatible_likelihoods_are_not_compared():
    bad = _fake(-2950., 20, n=499)                      # different observation count
    rows = [summarize_fit("1PL", _fake(-3000., 10), 500), summarize_fit("2PL", bad, 500)]
    assert rows[1]["reason"].startswith("Karşılaştırılamaz")
    other = _fake(-2950., 20); other.aic += 5           # library AIC with another definition
    assert summarize_fit("2PL", other, 500)["reason"].startswith("Karşılaştırılamaz")
    none = [summarize_fit(m, None, 500) for m in MODELS]
    assert "savunulabilir bir tercih edilen model seçilemez" in interpretation(none)


def test_interpretation_follows_results():
    a = interpretation(_rows([-3000., -2950., -2948., -2900.]))
    b = interpretation(_rows([-3000., -2999.5, -2999., -2900.]))
    assert "en küçük BIC değeri 2PL" in a and "en küçük BIC değeri 1PL" in b and a != b
    assert "karmaşıklığı karşılamamaktadır" in b


def test_page_one_click_one_table_one_paragraph_and_cache():
    pytest.importorskip("mirt")
    from streamlit.testing.v1 import AppTest
    from tests.test_reproducible_page_reruns import _state
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=900)
    for key, value in _state("irt_006", "MTK: Karşılaştırma").items():
        at.session_state[key] = value
    at.run()
    assert not at.exception and "irt_model_comparison" not in at.session_state
    at.button(key="irt_compare_run").click().run()
    assert not at.exception
    saved = at.session_state["irt_model_comparison"]
    assert [r["Model"] for r in saved["rows"]] == list(MODELS)
    tables = [d.value for d in at.dataframe if list(d.value.columns)[:3] == ["Model", "k", "LogLik"]]
    assert len(tables) == 1 and list(tables[0]["Model"]) == list(MODELS)
    assert sum("Model Seçiminin Değerlendirilmesi" in m.value for m in at.markdown) == 1
    calls = at.session_state["_repro_history"].manifest()["calls"]
    assert [c["function"] for c in calls].count("fit_python_mirt") == 4
    at.run()                                            # unrelated rerun: no refit, same stored result
    assert at.session_state["irt_model_comparison"] is saved
    at.session_state["item_cols"] = list(at.session_state["item_cols"])[:-1]
    at.run()
    assert "irt_model_comparison" not in at.session_state  # changed data invalidates the cache
