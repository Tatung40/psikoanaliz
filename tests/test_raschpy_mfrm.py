"""Experimental RaschPy MFRM: 8 models, independent identities, evidence gates, exports, UI, R/TAM untouched."""
import io, json, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("raschpy")
import core.raschpy_mfrm as RP
from core.mfrm import MFRMError

ROOT = Path(__file__).resolve().parents[1]; EX = ROOT / "examples/mfrm"
EV = json.loads((ROOT / "validation/raschpy_mfrm/evidence.json").read_text())
FIT = lambda name, model="global", **kw: RP.fit(pd.read_csv(EX / f"{name}.csv"), "person_id", "item_id", "rater_id", "score", model, **kw)


@pytest.mark.parametrize("model", list(RP.MODELS))
def test_all_models_fit_and_pass_independent_identities(model):
    r = FIT("unbalanced_linked", model); i = r["identity"]
    assert "error" not in i and r["seconds"] < 10
    assert i["V1_expected_score_max_diff"] <= 1e-12 and i["V2_wle_max_diff"] <= 1e-4 and i["V3_fit_max_diff"] <= 1e-6
    if model in RP.STRETCH: assert i["V1b_stretch_reconstruction_max_diff"] <= 1e-12
    assert {"Estimate", "SE"} <= set(r["items"].columns) and len(r["rater_params"])


def test_global_recovers_generating_values_and_matches_archived_estimate():
    r = FIT("balanced_3facet"); t = json.loads((EX / "fixtures.json").read_text())["fixtures"]["balanced_3facet"]["truth"]
    for k, v in t["raters"].items():
        assert abs(r["_m"].raters_global[k] - v) <= max(3 * r["_m"].rater_se_global[k], .15)
    assert abs(r["_m"].raters_global["P3"] - 0.866293) < 1e-5                      # deterministic: matches the discovery probe


def test_evidence_statuses_are_backed_and_never_certified():
    st = {m: v["parameters"]["status"] for m, v in EV["models"].items()}
    assert set(st) == set(RP.MODELS) and "CERTIFIED" not in st.values()
    cmp = pd.read_csv(ROOT / "validation/raschpy_mfrm/comparison.csv")
    for m, s in st.items():                                           # PARTIAL only when every recovery row passed
        rec = cmp[(cmp.model == m) & cmp.check.isin(["V4", "V5"])]
        assert (s == "PARTIAL") == bool(rec.ok.astype(bool).all())
    assert all(v["se"]["status"] == "NOT CERTIFIED" and v["model_selection"]["status"] == "NOT CERTIFIED" for v in EV["models"].values())
    assert all(e["failed_closed"] for e in EV["edge_cases"] if e["case"] not in ("sparse_category", "missing_scores"))


@pytest.mark.parametrize("name,match", [("disconnected", "bağlantılı değil"), ("invalid_duplicate_missing", "birden fazla puan"),
                                        ("four_facet", "birden fazla puan")])
def test_fail_closed_inputs(name, match):
    with pytest.raises(MFRMError, match=match):
        FIT(name)


def test_unsupported_features_are_declared_not_faked():
    assert {"andersen_lr_test", "extra_facets"} <= set(RP.UNSUPPORTED)
    with pytest.raises(MFRMError):
        RP.fit(pd.read_csv(EX / "balanced_3facet.csv"), "person_id", "item_id", "rater_id", "score", model="facets4")


def test_bootstrap_category_stats_and_model_selection():
    r = FIT("balanced_3facet", se_method="bootstrap", n_boot=30)
    assert r["category_stats"] is not None and "Disordered" in r["category_stats"].columns
    assert FIT("balanced_3facet")["category_stats"] is None              # sandwich: RaschPy category widths unavailable
    ms = RP.model_selection(r, "AIC"); assert set(ms.Model) == set(RP.MODELS)


def test_drf_with_person_covariate():
    df = pd.read_csv(EX / "balanced_3facet.csv"); df["grup"] = np.where(df.person_id.str[-1].astype(int) % 2, "A", "B")
    r = RP.fit(df, "person_id", "item_id", "rater_id", "score")
    tabs = RP.group_test(r, df, "person_id", "grup", "drf", n_boot=20); assert isinstance(tabs["tablo"], pd.DataFrame)


def test_exports_complete_and_pseudonymous():
    r = FIT("interaction_bias", "items"); files = RP.exports(r, {"source": "test"})
    pngs = [n for n in files if n.endswith(".png")]
    assert "wright_haritasi.png" in pngs and len(pngs) >= 6
    assert {"raschpy_mfrm.xlsx", "raschpy_mfrm_raporu.docx", "raschpy_mfrm.zip", "Kanıt durumu.csv"} <= set(files)
    persons = pd.read_csv(io.BytesIO(files["Bireyler.csv"])); assert persons.iloc[0, 0].startswith("Kişi_")
    from docx import Document
    text = "\n".join(p.text for p in Document(io.BytesIO(files["raschpy_mfrm_raporu.docx"])).paragraphs)
    assert "eşdeğer değildir" in text and "RCL" not in text or "bileşik olabilirlik" in text
    assert len(zipfile.ZipFile(io.BytesIO(files["raschpy_mfrm.zip"])).namelist()) == len(files) - 1


def test_page_runs_example_and_isolates_sessions():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string("from ui.raschpy_mfrm_view import render\nrender()", default_timeout=120)
    at.run(); assert not at.exception
    at.button(key="rpm_load").click().run(); at.selectbox(key="rpm_model").set_value("matrix").run()
    at.button(key="rpm_run").click().run(); assert not at.exception
    assert len(at.session_state["rpm_results"]) == 1 and any("Kestirim tamamlandı" in s.value for s in at.success)
    other = AppTest.from_string("from ui.raschpy_mfrm_view import render\nrender()", default_timeout=60); other.run()
    assert "rpm_results" not in other.session_state                         # separate session: no shared fits
    at.selectbox(key="rpm_example").set_value("disconnected").run(); at.button(key="rpm_load").click().run()
    assert at.button(key="rpm_run").disabled and any("bağlantılı değil" in e.value for e in at.error)


def test_tam_engine_files_untouched():
    import subprocess
    out = subprocess.run(["git", "diff", "--name-only", "origin/main", "--", "core/mfrm.py", "ui/mfrm_view.py", "scripts/mfrm_tam.R",
                          "validation/mfrm"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode: pytest.skip("git history unavailable")
    assert out.stdout.strip() == ""
