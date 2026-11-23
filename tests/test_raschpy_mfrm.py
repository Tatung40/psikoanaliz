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
    assert i["V2b_csem_max_diff"] <= 1e-4 and i["V13_separation_max_diff"] <= 1e-6
    if model in RP.STRETCH: assert i["V1b_stretch_reconstruction_max_diff"] <= 1e-12
    assert {"Estimate", "SE"} <= set(r["items"].columns) and len(r["rater_params"])


def test_global_recovers_generating_values_and_matches_archived_estimate():
    r = FIT("balanced_3facet"); t = json.loads((EX / "fixtures.json").read_text())["fixtures"]["balanced_3facet"]["truth"]
    for k, v in t["raters"].items():
        assert abs(r["_m"].raters_global[k] - v) <= max(3 * r["_m"].rater_se_global[k], .15)
    assert abs(r["_m"].raters_global["P3"] - 0.866293) < 1e-5                      # deterministic: matches the discovery probe


def test_evidence_statuses_are_backed_by_gates():
    st = {m: v["parameters"]["status"] for m, v in EV["models"].items()}; assert set(st) == set(RP.MODELS)
    cmp = pd.read_csv(ROOT / "validation/raschpy_mfrm/comparison.csv")
    V18 = json.loads((ROOT / "validation/raschpy_mfrm/familywise_v18.json").read_text())["results"]
    for m, s in st.items():                       # CERTIFIED = identities + V8 + V18 family-wise (both designs)
        rec = cmp[(cmp.model == m) & cmp.check.isin(["V4", "V5"])]; v8 = cmp[(cmp.model == m) & cmp.check.str.startswith("V8_") & cmp.ok.notna()]
        hist = bool(rec.ok.astype(bool).all() and v8.ok.astype(bool).all())
        assert EV["models"][m]["parameters"]["historical_rule_status"] == ("CERTIFIED" if hist else "NOT CERTIFIED")  # history kept
        assert len(v8) == 4 and (s == "CERTIFIED") == bool(v8.ok.astype(bool).all() and V18[m]["pass_both_designs"])
        if hist and not V18[m]["pass_both_designs"]: assert s == "PARTIAL"                 # downgrade rule
        if not hist and not V18[m]["pass_both_designs"]: assert s == "NOT CERTIFIED"
    disc = cmp[cmp.check == "V8_discrimination"].value; assert disc.min() > 1e-3     # the two configurations differ
    for m, v in EV["models"].items():
        assert v["model_selection"]["status"] == "UNSUPPORTED" and not hasattr(RP, "model_selection")
        assert (v["se"]["status"] == "PARTIAL") == any(RP.se_reliable(m, g) for g in ("rater", "items", "thresholds"))
        assert v["test_stats"]["status"] in ("PARTIAL", "NOT CERTIFIED") and v["test_stats"]["max_diff"] <= 1e-6
    for m in RP.BOOT_MODELS:                                     # historical V11 failure stays recorded as failed
        v11 = cmp[(cmp.model == m) & (cmp.check == "V11")]
        assert len(v11) and (not v11.ok.astype(bool).all())
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


def test_bootstrap_se_never_shown_as_validated_and_widths_derived():
    r = FIT("balanced_3facet", se_method="bootstrap", n_boot=30); T = RP.tables(r)
    assert "SE" not in T["Maddeler"] and "SE" not in T["Eşikler"] and not any("genişliği (bootstrap)" in k for k in T)
    thr = dict((k, d) for k, _, d, _ in RP.academic_tables(r))["thresholds"]
    tau = r["thresholds"]["Estimate"].to_numpy(float); w = thr["Kategori genişliği (τ_k − τ_k−1)"].to_numpy(float)
    assert np.isnan(w[0]) and np.allclose(w[1:], np.diff(tau))
    raschpy_w = r["category_stats"]["Estimate"].to_numpy(float)                        # RaschPy's own widths (rounded)
    assert np.allclose(raschpy_w, np.round(np.diff(tau), 3), atol=1e-3)


def test_category_coding_from_one_fails_with_clear_message():
    d = pd.read_csv(EX / "balanced_3facet.csv"); d["score"] += 1
    with pytest.raises(MFRMError, match="0'dan başlamasını"):
        RP.fit(d, "person_id", "item_id", "rater_id", "score")


def test_reliability_definitions_reported_separately():
    r = FIT("balanced_3facet"); t = dict((k, d) for k, _, d, _ in RP.academic_tables(r))["test"]
    rows = t.set_index("Yüzey")
    assert {"Bireyler (RaschPy, gerçek SEM)", "Bireyler (Wright–Masters, model SEM)"} <= set(rows.index)
    p = r["persons"]; ne = p.index[(p.Score > 0) & (p.Score < p["Max score"])]; m = r["_m"]   # unrounded values
    th = m.persons_global[ne]; v = np.var(th); mse = (m.csem_vector_global[ne] ** 2).mean(); g = np.sqrt((v - mse) / mse)
    assert abs(rows.loc["Bireyler (Wright–Masters, model SEM)", "Güvenirlik"] - g ** 2 / (1 + g ** 2)) < 1e-6


def test_drf_with_person_covariate():
    df = pd.read_csv(EX / "balanced_3facet.csv"); df["grup"] = np.where(df.person_id.str[-1].astype(int) % 2, "A", "B")
    r = RP.fit(df, "person_id", "item_id", "rater_id", "score")
    tabs = RP.group_test(r, df, "person_id", "grup", "drf", n_boot=20); assert isinstance(tabs["tablo"], pd.DataFrame)


def test_unreliable_uncertainty_is_hidden_and_cell_bootstrap_gated():
    r = FIT("balanced_3facet", "matrix"); T = RP.tables(r)
    for name, g in (("Maddeler", "items"), ("Eşikler", "thresholds"), ("Puanlayıcı parametreleri", "rater")):
        assert ("SE" in T[name]) == RP.se_reliable("matrix", g)                     # SE shown only with coverage evidence
    with pytest.raises(MFRMError): RP.cell_bootstrap(r, B=5)                         # not validated for matrix
    c = FIT("balanced_3facet", "centrality"); c["cell_bootstrap"] = RP.cell_bootstrap(c, B=10)
    b = RP.tables(c)["Puanlayıcı hücreleri (bootstrap SE/GA)"]
    assert (b["%95 GA alt"] < b.Kestirim).all() and (b["Bootstrap SE"] > 0).all()
    assert ("SE" in RP.tables(c)["Puanlayıcı parametreleri"]) == RP.se_reliable("centrality", "rater")


def test_exports_complete_and_pseudonymous():
    r = FIT("interaction_bias", "items"); files = RP.exports(r, {"source": "test"})
    pngs = [n for n in files if n.endswith(".png")]
    assert "wright_haritasi.png" in pngs and len(pngs) >= 6
    assert {"raschpy_mfrm.xlsx", "raschpy_mfrm_raporu.docx", "raschpy_mfrm.zip", "Kanıt durumu.csv"} <= set(files)
    persons = pd.read_csv(io.BytesIO(files["Bireyler.csv"])); assert persons.iloc[0, 0].startswith("Kişi_")
    from docx import Document
    text = "\n".join(p.text for p in Document(io.BytesIO(files["raschpy_mfrm_raporu.docx"])).paragraphs)
    assert "eşdeğer değildir" in text and "kategori standart hataları için kapsama kanıtı" in text.lower()
    assert len(zipfile.ZipFile(io.BytesIO(files["raschpy_mfrm.zip"])).namelist()) == len(files) - 1


@pytest.mark.parametrize("model", ["global", "bistretch"])
def test_academic_word_report_is_clean_numbered_and_honest(model):
    from docx import Document
    r = FIT("balanced_3facet", model); figs = RP.plots(r, fast=True)
    doc = Document(io.BytesIO(RP.docx_report(r, figs, {"source": {"sha": "ab" * 32}})))
    text = "\n".join(p.text for p in doc.paragraphs)
    for leak in ("<div", "</", "{'", "core.", "raschpy.MFRM(", "calibrate(", "V1_", "_max_diff", "Traceback", "unsafe_allow_html"):
        assert leak not in text, leak
    titles = [doc.paragraphs[i + 1].text for i, p in enumerate(doc.paragraphs) if p.text.startswith("Tablo ")]
    assert titles[:3] == ["Madde Parametreleri", "Eşik Parametreleri", "Puanlayıcı Katılıkları"]
    assert "Birey Ölçüleri" in titles and "Çıktıların Bilimsel Durumu" in titles
    nums = [int(p.text.split()[1]) for p in doc.paragraphs if p.text.startswith("Tablo ")]; assert nums == list(range(1, len(nums) + 1))
    assert len(doc.inline_shapes) == len(figs) and sum(p.text.startswith("Şekil ") for p in doc.paragraphs) == len(figs)
    for h in ("1. Amaç ve Kapsam", "2. Yöntem", "2.1 Veri ve Örneklem", "2.3 Kestirim", "3. Bulgular", "Bilimsel Durum ve Sınırlılıklar", "Tekrarlanabilirlik"):
        assert any(p.text.endswith(h) or p.text == h for p in doc.paragraphs), h
    persons = next(t for t, p in zip(doc.tables, [x for x in titles]) if p == "Birey Ölçüleri")
    assert len(persons.rows) == 16 and "tam tablosu Excel/CSV" in text                 # 15 shown, 300 in Excel/CSV
    assert "Uç puanlı" in text and "Eksik puanlar" in text and "GENERATED" in text
    status = RP.status(model, "parameters")
    assert (f"Bilimsel durum: {RP.STATUS_TR[status]}" in text) and (("parametreleri NOT CERTIFIED" in text) == (status == "NOT CERTIFIED"))
    assert ("SE" in [c.text for c in doc.tables[0].rows[0].cells]) == RP.se_reliable(model, "items")   # SE only with evidence
    assert not any(c.text in ("nan", "None") for t in doc.tables for row in t.rows for c in row.cells)


def test_excel_keeps_full_person_table():
    r = FIT("balanced_3facet"); files = RP.exports(r, {}, figs=RP.plots(r, fast=True))
    assert len(pd.read_excel(io.BytesIO(files["raschpy_mfrm.xlsx"]), sheet_name="Bireyler")) == 300


def test_page_runs_example_and_isolates_sessions():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string("from ui.raschpy_mfrm_view import render\nrender()", default_timeout=120)
    at.run(); assert not at.exception
    at.button(key="rpm_load").click().run(); at.selectbox(key="rpm_model").set_value("matrix").run()
    at.button(key="rpm_run").click().run(); assert not at.exception
    assert len(at.session_state["rpm_results"]) == 1 and any("Kestirim tamamlandı" in s.value for s in at.success)
    assert [t.label for t in at.tabs] == ["Genel Bakış", "Parametreler", "Bireyler", "Grafikler", "Tanılar", "İndir"]
    assert any(e.label == "İleri Düzey / Teknik Ayrıntılar" and not e.proto.expanded for e in at.expander)
    assert not at.json and any("Puanlayıcı Katılıkları" in m.value for m in at.markdown)
    assert {m.label for m in at.metric} >= {"Birey (N)", "Madde", "Puanlayıcı", "Model", "Parametre durumu"}
    other = AppTest.from_string("from ui.raschpy_mfrm_view import render\nrender()", default_timeout=60); other.run()
    assert "rpm_results" not in other.session_state                         # separate session: no shared fits
    at.selectbox(key="rpm_example").set_value("disconnected").run(); at.button(key="rpm_load").click().run()
    assert at.button(key="rpm_run").disabled and any("bağlantılı değil" in e.value for e in at.error)


def test_independent_estimator_reproduces_default_fit():
    from core.mfrm import prepare
    import validation.raschpy_mfrm.independent as V
    d, _ = prepare(pd.read_csv(EX / "unbalanced_linked.csv"), "person_id", "item_id", "rater_id", "score")
    for model in ("global", "bistretch"):
        assert V.compare(d, model, 3, regularised=True)[0] <= 1e-4


def test_tam_engine_files_untouched():
    import subprocess
    out = subprocess.run(["git", "diff", "--name-only", "origin/main", "--", "core/mfrm.py", "ui/mfrm_view.py", "scripts/mfrm_tam.R",
                          "validation/mfrm"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode: pytest.skip("git history unavailable")
    assert out.stdout.strip() == ""


def test_academic_category_counts_no_duplicate_total():
    """A RaschPy Total row must not be counted again in the report."""
    res = FIT("balanced_3facet")
    table = next(t for key, _, t, _ in RP.academic_tables(res, max_persons=15) if key == "category_counts")
    assert "Total" not in table.columns
    assert "Missing" not in table.columns
    assert "Toplam" in table.columns
    item_cols = [c for c in table.columns if c not in ("Kategori", "Toplam")]
    assert len(item_cols) == 4
    assert (table["Toplam"] == table[item_cols].sum(axis=1)).all()
    assert int(table["Toplam"].sum()) == 3600


def test_exported_values_match_fitted_model_and_screen():
    from docx import Document
    r = FIT("unbalanced_linked", "items"); files = RP.exports(r, {}, figs={})
    doc = Document(io.BytesIO(files["raschpy_mfrm_raporu.docx"])); t1 = doc.tables[0]
    word = {row.cells[0].text: float(row.cells[1].text) for row in t1.rows[1:]}
    assert word == {k: round(float(v), 3) for k, v in r["_m"].items_items.items()}         # Word = fitted (3 dp)
    xl = pd.read_excel(io.BytesIO(files["raschpy_mfrm.xlsx"]), sheet_name="Maddeler").set_index("Öğe")
    assert np.allclose(xl["Estimate"].to_numpy(float), r["items"]["Estimate"].to_numpy(float))
    cc = pd.read_excel(io.BytesIO(files["raschpy_mfrm.xlsx"]), sheet_name="Kategori sıklıkları")
    screen = dict((k, d) for k, _, d, _ in RP.academic_tables(r))["category_counts"]
    assert int(screen["Toplam"].sum()) == len(r["_d"])                                       # every score counted once
    cc = cc[cc["Öğe"].astype(str) != "Total"]                                                   # RaschPy's labelled total row
    assert int(cc.select_dtypes("number").drop(columns=["Total", "Missing"], errors="ignore").to_numpy().sum()) == len(r["_d"])


@pytest.mark.parametrize("model", list(RP.MODELS))
def test_page_all_models_render_with_consistent_status(model):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string("from ui.raschpy_mfrm_view import render\nrender()", default_timeout=180)
    at.run(); at.button(key="rpm_load").click().run(); at.selectbox(key="rpm_model").set_value(model).run()
    at.button(key="rpm_run").click().run(); assert not at.exception
    metric = {m.label: m.value for m in at.metric}["Parametre durumu"]
    assert metric == RP.status(model, "parameters")
    assert any(w for w in at.warning if "NOT CERTIFIED" in w.value) == (metric == "NOT CERTIFIED")
    assert not any("AIC" in b.label for b in at.button)


def test_global_bh_decision_and_uncalibrated_omnibus_label():
    df = pd.read_csv(EX / "balanced_3facet.csv"); df["grup"] = np.where(df.person_id.str[-1].astype(int) % 2, "A", "B")
    r = RP.fit(df, "person_id", "item_id", "rater_id", "score")
    tabs = RP.group_test(r, df, "person_id", "grup", "dif")
    g = tabs["global_bh"].iloc[0]; pmin = float(tabs["tablo"]["p (corrected)"].min())
    assert g["En küçük BH-düzeltilmiş p"] == pmin and (g["Karar (α = 0.05)"] == "global boş hipotez reddedildi") == (pmin < .05)
    v = EV["models"]["global"]["dif"]["v15"] or {}
    assert RP.group_part_validated("global", "dif", "omnibus") == bool(v.get("gate_omnibus"))
    assert RP.group_part_validated("global", "dif", "global_bh") == bool(v.get("gate_global_bh"))
    assert not RP.group_part_validated("matrix", "dif", "global_bh")                     # scope: global model only


def test_v17_holdout_diagnosis_recorded():
    h = json.loads((ROOT / "validation/raschpy_mfrm/holdout_v17.json").read_text())["results"]
    for m in RP.BOOT_MODELS:
        for fx in ("balanced_3facet", "unbalanced_linked"):
            r = h[f"{m}:{fx}"]; assert r["replications"] == 200 and r["fit_failures"] == 0 and r["bias_gate"]
        assert EV["models"][m]["parameters"]["historical_rule_status"] == "NOT CERTIFIED"


def test_v18_familywise_results_complete_and_equal_for_all_models():
    r = json.loads((ROOT / "validation/raschpy_mfrm/familywise_v18.json").read_text())
    assert r["reps"] == 200 and set(r["results"]) == set(RP.MODELS)
    for m, v in r["results"].items():
        for d in ("balanced", "linked"):
            x = v[d]; assert x["fit_failures"] == 0 and x["confirmatory_replications"] == 100
            assert x["pass"] == all(x[k] for k in ("gate_bias", "gate_se_ratio", "gate_coverage", "gate_familywise"))
