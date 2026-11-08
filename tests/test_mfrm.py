"""MFRM module: fail-closed data checks, Python identities on archived TAM output, evidence gates, exports, UI, live R."""
import io, json, os, shutil, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import core.mfrm as M

ROOT = Path(__file__).resolve().parents[1]; EX = ROOT / "examples/mfrm"; ARCH = ROOT / "validation/mfrm/tam_output"
EVID = json.loads((ROOT / "validation/mfrm/evidence.json").read_text())["outputs"]


def archived(name):
    def engine(d, facets, interaction):
        o = ARCH / name
        raw = {"xsi": pd.read_csv(o / "parameters.csv"), "persons": pd.read_csv(o / "persons.csv", dtype={"person": str}),
               "summary": pd.read_csv(o / "summary.csv").set_index("key").value.to_dict(), "versions": (o / "versions.txt").read_text().splitlines()}
        if interaction and (o / "interaction.csv").exists():
            raw["interaction"] = pd.read_csv(o / "interaction.csv")
        return raw
    return engine


def load(name, **kw):
    return M.analyze(pd.read_csv(EX / f"{name}.csv"), "person_id", "item_id", "rater_id", "score", engine=archived(name), **kw)


@pytest.mark.parametrize("name,match", [("disconnected", "bağlantılı değil"), ("invalid_duplicate_missing", "birden fazla puan")])
def test_invalid_designs_fail_closed(name, match):
    with pytest.raises(M.MFRMError, match=match):
        M.prepare(pd.read_csv(EX / f"{name}.csv"), "person_id", "item_id", "rater_id", "score")


def test_data_rules_are_explicit_and_never_repair():
    d = pd.read_csv(EX / "balanced_3facet.csv")
    x = d.copy(); x.loc[0, "score"] = np.nan; _, rep = M.prepare(x, "person_id", "item_id", "rater_id", "score")
    assert rep["missing_scores_excluded"] == 1 and rep["n_rows"] == len(d) - 1
    for change, match in ((lambda z: z.assign(score=z.score + .5), "tam sayı"), (lambda z: z.assign(score=z.score.replace(1, 2)), "Ara kategori"),
                          (lambda z: z[z.rater_id == "P1"], "en az iki düzey"), (lambda z: z.assign(person_id=None), "boş kimlik"),
                          (lambda z: z.astype({"score": object}).assign(score=["x"] + list(z.score[1:])), "sayı olmayan")):
        with pytest.raises(M.MFRMError, match=match):
            M.prepare(change(d.copy()), "person_id", "item_id", "rater_id", "score")
    with pytest.raises(M.MFRMError, match="Dört ve daha fazla"):
        M.analyze(pd.read_csv(EX / "four_facet.csv"), "person_id", "item_id", "rater_id", "score", ("task_id",))
    full = pd.read_csv(EX / "sparse_category.csv")                    # 88 / 2400 in the top category: not sparse (>= 10)
    assert not M.prepare(full, "person_id", "item_id", "rater_id", "score")[1]["sparse_categories"]
    few = full[full.person_id.isin(full.person_id.unique()[:20])]       # Linacre: < 10 observations in a category
    assert M.prepare(few, "person_id", "item_id", "rater_id", "score")[1]["sparse_categories"] == [3]


def test_python_identities_reproduce_archived_tam_output():
    r = load("balanced_3facet")
    assert r["identity"]["loglik"] <= 1e-5 and r["identity"]["wle"] <= 1e-4 and r["identity"]["eap"] <= 1e-5
    el = r["elements"]; assert abs(el.loc[el.facet == "rater", "Ölçü (logit)"].sum()) < 1e-6        # sum-to-zero identification
    assert list(el.sort_values("Ölçü (logit)").query("facet == 'rater'").Düzey) == ["P1", "P2", "P3"]   # lenient → severe
    ref = pd.read_csv(ROOT / "validation/mfrm/reference.csv")
    mm = ref[(ref.fixture == "balanced_3facet") & (ref.engine == "mfrmr") & (ref.kind == "element") & (ref.key == "Estimate")]
    for e in el.itertuples():
        q = mm[(mm.level == e.Düzey) & (mm.facet == {"item": "item_id", "rater": "rater_id"}[e.facet])].value.item()
        assert abs(e._4 - q) <= 5e-3


def test_mapping_gate_withholds_everything_on_mismatch():
    bad = archived("balanced_3facet")
    def engine(*a):
        raw = bad(*a); raw["summary"]["loglik"] += 1.0; return raw
    with pytest.raises(M.MFRMError, match="eşlemesi"):
        M.analyze(pd.read_csv(EX / "balanced_3facet.csv"), "person_id", "item_id", "rater_id", "score", engine=engine)
    def nonconv(*a):
        raw = bad(*a); raw["summary"]["converged"] = 0.; return raw
    with pytest.raises(M.MFRMError, match="yakınsamadı"):
        M.analyze(pd.read_csv(EX / "balanced_3facet.csv"), "person_id", "item_id", "rater_id", "score", engine=nonconv)


def test_evidence_labels_and_withheld_outputs():
    certified = {k for k, v in EVID.items() if v.get("status") == "CERTIFIED"}
    assert certified == {"facet_measures", "thresholds", "model_fit", "latent", "rater_ranking"}
    assert all(EVID[k]["max_diff"] <= 5e-3 for k in certified - {"model_fit"}) and EVID["model_fit"]["max_diff"] <= .05
    assert EVID["fit_zstd"]["status"] == EVID["interaction_estimates"]["status"] == EVID["interaction_lr"]["status"] == "NOT CERTIFIED"
    r = load("interaction_bias", interaction=True); view = M.present(r)
    assert "Etkileşim LR" not in view["tables"] and any("yakınsamadı" in n for n in view["notes"])
    assert not any("ZSTD" in c for t in view["tables"].values() for c in t.columns)


def test_pseudonymous_ids_by_default_and_exports_match_display():
    from ui.mfrm_view import _exports
    r = load("balanced_3facet"); view = M.present(r)
    assert view["tables"]["Bireyler"]["Birey"].str.startswith("Kişi_").all()
    files = _exports(r, view, {"show_ids": False})
    for name, t in view["tables"].items():
        assert pd.read_csv(io.BytesIO(files[f"{name}.csv"]), encoding="utf-8-sig").shape == t.shape
    assert not any("S0000" in files[f].decode("utf-8-sig") for f in files if f.endswith(".csv"))
    names = zipfile.ZipFile(io.BytesIO(files["mfrm.zip"])).namelist()
    assert {"mfrm_raporu.docx", "wright_haritasi.png", "kategori_egrileri.png", "yuzey_olculeri.png"} <= set(names)
    from docx import Document
    text = "\n".join(p.text for p in Document(io.BytesIO(files["mfrm_raporu.docx"])).paragraphs)
    assert "FACETS" in text and "eşdeğer değildir" in text and "TAM" in text
    shown = load("balanced_3facet", show_ids=True)
    assert "S0000" in M.present(shown)["tables"]["Bireyler"]["Birey"].tolist()


def test_page_workflow_with_archived_engine(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(M, "engine_available", lambda: (True, ""))
    monkeypatch.setattr(M, "run_tam", archived("balanced_3facet"))
    at = AppTest.from_string("from ui.mfrm_view import render\nrender()", default_timeout=180)
    at.run(); assert not at.exception
    at.button(key="mfrm_load").click().run(); at.button(key="mfrm_run").click().run(); assert not at.exception
    assert "result" in at.session_state["mfrm_result"]
    at.selectbox(key="mfrm_example").set_value("unbalanced_linked").run(); at.button(key="mfrm_load").click().run()
    assert "mfrm_result" not in at.session_state                                   # stale result cleared


def test_live_tam_engine_matches_archive_when_available():
    ok, why = M.engine_available()
    if not ok:
        if os.getenv("REQUIRE_MFRM_R") == "1": pytest.fail(why)
        pytest.skip("NOT EXECUTED: " + why)
    live = M.analyze(pd.read_csv(EX / "balanced_3facet.csv"), "person_id", "item_id", "rater_id", "score")
    arch = load("balanced_3facet")
    assert np.allclose(live["elements"]["Ölçü (logit)"], arch["elements"]["Ölçü (logit)"], atol=1e-8)
    assert abs(live["model"]["logLik"] - arch["model"]["logLik"]) <= 1e-8


def test_reference_rerun_matches_archive_when_available(tmp_path):
    import subprocess
    rs = shutil.which("Rscript")
    probe = subprocess.run([rs, "-e", "library(mfrmr); library(rasch)"], capture_output=True) if rs else None
    if not rs or probe.returncode != 0:
        if os.getenv("REQUIRE_MFRM_R") == "1": pytest.fail("mfrmr/rasch required")
        pytest.skip("NOT EXECUTED: R mfrmr/rasch unavailable")
    out = tmp_path / "reference.csv"
    run = subprocess.run([rs, str(ROOT / "validation/mfrm/reference.R"), str(EX), str(out)], capture_output=True, text=True, timeout=1500)
    assert run.returncode == 0, run.stderr[-1500:]
    new, old = pd.read_csv(out), pd.read_csv(ROOT / "validation/mfrm/reference.csv")
    assert len(new) == len(old) and np.allclose(new.value, old.value, atol=1e-6, equal_nan=True)


def test_run_button_always_visible_disabled_without_engine(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(M, "engine_available", lambda: (False, "Rscript bulunamadı."))
    at = AppTest.from_string("from ui.mfrm_view import render\nrender()", default_timeout=120)
    at.run(); at.button(key="mfrm_load").click().run(); assert not at.exception
    btn = at.button(key="mfrm_run"); assert btn.disabled
    assert any("MFRM motoru kullanılamıyor" in e.value for e in at.error) and any("Dockerfile" in i.value for i in at.info)
    assert "mfrm_result" not in at.session_state                                    # disabled: no estimate, no fake result
    at.selectbox(key="mfrm_example").set_value("disconnected").run(); at.button(key="mfrm_load").click().run()
    assert at.button(key="mfrm_run").disabled and any("bağlantılı değil" in e.value for e in at.error)


@pytest.mark.parametrize("name", ["balanced_3facet", "unbalanced_linked", "interaction_bias"])
def test_three_examples_end_to_end_on_page(monkeypatch, name):
    """Real TAM when available (REQUIRE_MFRM_R=1 in the native R job); archived TAM output otherwise."""
    from streamlit.testing.v1 import AppTest
    ok, why = M.engine_available()
    if not ok:
        if os.getenv("REQUIRE_MFRM_R") == "1": pytest.fail(why)
        if name not in {"balanced_3facet", "interaction_bias"}: pytest.skip("NOT EXECUTED: no engine and no archived TAM output")
        monkeypatch.setattr(M, "engine_available", lambda: (True, "")); monkeypatch.setattr(M, "run_tam", archived(name))
    at = AppTest.from_string("from ui.mfrm_view import render\nrender()", default_timeout=600)
    at.run(); at.selectbox(key="mfrm_example").set_value(name).run(); at.button(key="mfrm_load").click().run()
    if name == "interaction_bias":
        at.checkbox(key="mfrm_interaction").check().run()
    assert not at.button(key="mfrm_run").disabled
    at.button(key="mfrm_run").click().run(); assert not at.exception
    saved = at.session_state["mfrm_result"]; assert "result" in saved, saved.get("error")
    r = saved["result"]; el = r["elements"]
    ref = pd.read_csv(ROOT / "validation/mfrm/reference.csv")
    mm = ref[(ref.fixture == name) & (ref.engine == "mfrmr") & (ref.kind == "element") & (ref.key == "Estimate")]
    for e in el.itertuples():
        q = mm[(mm.level == e.Düzey) & (mm.facet == {"item": "item_id", "rater": "rater_id"}[e.facet])].value.item()
        assert abs(e._4 - q) <= 5e-3                                                      # preregistered tolerance
    assert r["identity"]["loglik"] <= 1e-5
    from ui.mfrm_view import _exports
    view = M.present(r); files = _exports(r, view, {"example": name})
    assert {"wright_haritasi.png", "kategori_egrileri.png", "mfrm_sonuclari.xlsx", "mfrm_raporu.docx", "mfrm.zip"} <= set(files)
    assert all(f"{t}.csv" in files for t in view["tables"])
    assert len(at.dataframe) >= len(view["tables"])
