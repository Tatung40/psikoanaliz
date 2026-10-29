"""3PL/4PL usage policy and MAP adapter option (pre-registered evidence, validation/evidence/p1_3pl4pl)."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from core.irt_binary_usage_policy import EVIDENCE, usage_status

ROOT = Path(__file__).resolve().parents[1]


def test_policy_matches_preregistered_evidence_cells():
    summary = json.loads((ROOT / EVIDENCE).read_text(encoding="utf-8"))["summary"]
    for key, cell in summary.items():
        model, n, k, est, theta = key.split("|")
        if theta != "normal" or est not in ("ML21", "MAP"):
            continue
        status, _ = usage_status(model, "ML" if est == "ML21" else "MAP", int(n[1:]), int(k[1:]))
        # "supported" must never be issued for a cell that failed the pre-registered criteria
        if status == "supported":
            assert cell["meets_preregistered_criteria"], key
        if cell["meets_preregistered_criteria"] and est == "MAP":
            assert status == "supported", key


def test_policy_messages_and_scope():
    assert usage_status("2PL", "ML", 500, 10)[0] == "not_applicable"
    assert usage_status("4PL", "MAP", 5000, 8)[0] == "not_supported"
    assert usage_status("3PL", "MAP", 800, 20)[0] == "not_supported"
    assert "N(0,1)" in usage_status("3PL", "MAP", 3000, 20)[1]
    with pytest.raises(ValueError):
        usage_status("3PL", "EAP", 3000, 20)


def test_adapter_map_option_uses_library_default_priors():
    pytest.importorskip("mirt")
    from core.irt_python_mirt_adapter import fit_python_mirt
    rng = np.random.default_rng(3); n, k = 1500, 12
    th = rng.standard_normal(n); a = np.linspace(.8, 1.6, k); b = np.linspace(-1.2, 1.2, k)
    x = (rng.random((n, k)) < .15 + .8 * expit(a * (th[:, None] - b))).astype(int)
    ml = fit_python_mirt(x, model="4PL"); mp = fit_python_mirt(x, model="4PL", estimation="MAP")
    assert ml.estimation == "ML" and mp.estimation == "MAP"
    a_map = mp.fit.coef()["discrimination"].to_numpy()
    assert (a_map < 4.99).all() and not np.allclose(a_map, ml.fit.coef()["discrimination"].to_numpy())
    with pytest.raises(ValueError):
        fit_python_mirt(x, model="2PL", estimation="MAP")
    with pytest.raises(ValueError):
        fit_python_mirt(x, model="3PL", estimation="Bayes")


@pytest.mark.parametrize("model", ["3PL", "4PL"])
def test_3pl_4pl_page_renders_curves_with_policy_columns(model):
    st = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("mirt")
    import pandas as pd
    rng = np.random.default_rng(9); n, k = 600, 12
    th = rng.standard_normal(n)
    x = (rng.random((n, k)) < .15 + .8 * expit(np.linspace(.8, 1.6, k) * (th[:, None] - np.linspace(-1, 1, k)))).astype(int)
    df = pd.DataFrame(x, columns=[f"M{j}" for j in range(k)])
    at = st.AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
    for key, value in (("raw_df", df), ("scored_df", df.copy()), ("item_cols", list(df.columns)), ("id_col", None),
                       ("group_col", None), ("is_dichotomous", True), ("nav_page", f"MTK: {model}")):
        at.session_state[key] = value
    at.run()
    at.button(key=f"py_mirt_run_{model}").click().run()
    assert not at.exception
    assert not any("eşleştirme hatası" in e.value for e in at.error)
    assert len(at.get("plotly_chart")) >= 3
    table = next(d.value for d in at.dataframe if "Sınırdaki parametre" in d.value.columns)
    assert set(table["Kestirim"]) == {"MAP"} and set(table["Kullanım durumu"]) == {"not_supported"}  # N = 600 < 1000


def test_evidence_note_is_model_specific_and_matches_stored_coverage():
    import json
    from pathlib import Path
    from core.irt_binary_usage_policy import evidence_note
    p6 = json.loads((Path(__file__).resolve().parents[1] / "validation/evidence/p6_binary_se/coverage.json").read_text())
    cov = p6["results"]
    assert f"{cov['1PL']['b']['coverage']:.3f}".lstrip("0") in evidence_note("1PL")[1]
    lo, hi = sorted(round(cov["2PL"][q]["coverage"], 3) for q in "ab")
    assert f"{lo:.3f}".lstrip("0") + "–" + f"{hi:.3f}".lstrip("0") in evidence_note("2PL")[1]
    assert [evidence_note(m)[0] for m in ("Rasch", "1PL", "2PL", "3PL", "4PL")] == ["info"] * 4 + ["warning"]
    assert evidence_note("1PL", common_discrimination=True)[0] == "warning"
    assert "NOT VERIFIED" in evidence_note("4PL")[1] and "araştırma amaçlı" not in evidence_note("2PL")[1]
