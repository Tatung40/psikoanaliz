"""Inter-rater module: registry, 30 example paths, invariances, round trips, fail-closed checks, R reference, UI."""
import hashlib, io, json, os, shutil, subprocess, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import core.rater_reliability as R
from validation.rater_reliability.compare import KEY, compare, load, tolerance

ROOT = Path(__file__).resolve().parents[1]; EX = ROOT / "examples/rater_reliability"
REG = json.loads((EX / "registry.json").read_text(encoding="utf-8"))
EVID = json.loads((ROOT / "validation/rater_reliability/evidence.json").read_text(encoding="utf-8"))["status"]


def test_registry_has_30_methods_each_with_a_documented_example():
    assert len(R.METHODS) == 30 and len(REG["examples"]) == 30
    assert {e["method"] for e in REG["examples"]} == set(R.METHODS)
    for e in REG["examples"]:
        assert hashlib.sha256((EX / e["file"]).read_bytes()).hexdigest() == e["sha256"]
        assert all(e["columns"].values()) and e["expected"] and e["scenario"] and e["missing"]
        assert R.METHODS[e["method"]]["example"] == e["id"]
    xl = pd.ExcelFile(EX / REG["workbook"])
    assert {e["sheet"] for e in REG["examples"]} <= set(xl.sheet_names) and "Aciklama" in xl.sheet_names


@pytest.mark.parametrize("example", REG["examples"], ids=lambda e: e["id"])
def test_every_example_runs_through_production_and_matches_independent_expected(example):
    data = load(example); out = R.run(example["method"], data)
    assert out["interpretation"] and out["status"].startswith("NUMERICALLY VALIDATED")
    failed = set(EVID[example["method"]]["failed_outputs"])
    for key, exp in example["expected"].items():
        name, get = KEY[key]
        if name in failed:
            assert name not in out["shown"]                     # unvalidated output is never displayed
            continue
        got = get(out["result"]); assert np.isfinite(got)
        assert abs(got - exp["value"]) <= tolerance(exp["source"], key, exp["value"], example["method"]), (key, got, exp)


def test_csv_and_xlsx_round_trip_give_identical_results():
    xl = pd.ExcelFile(EX / REG["workbook"])
    for e in REG["examples"]:
        a = R.compute(e["method"], load(e))
        df = xl.parse(e["sheet"]); c = e["config"]
        d = R.from_counts(df, c["id_col"], c["category_cols"]) if c["layout"] == "counts" else R.from_wide(df, c["id_col"], c["rater_cols"])
        b = R.compute(e["method"], d)
        for k in ("estimate", "chi2", "bias"):
            if k in a: assert abs(a[k] - b[k]) <= 1e-12, e["id"]


def test_wide_long_and_counts_transformations_are_exact():
    e = REG["examples"][[x["method"] for x in REG["examples"]].index("conger_kappa")]; wide = load(e)
    long = R.from_long(R.to_long(wide), "Birim_ID", "Puanlayici_ID", "Puan")
    shipped = REG["teaching"]["long"]["config"]
    from_file = R.from_long(pd.read_csv(EX / REG["teaching"]["long"]["file"]), shipped["unit_col"], shipped["rater_col"], shipped["score_col"])
    counts = R.from_counts(R.counts_from_wide(wide).reset_index(names="Birim_ID"), "Birim_ID", list(R.counts_from_wide(wide).columns))
    for m in ("fleiss_kappa", "gwet_ac1", "brennan_prediger", "percent_agreement"):
        v = R.compute(m, wide)["estimate"]
        assert abs(R.compute(m, long)["estimate"] - v) <= 1e-12 and abs(R.compute(m, from_file)["estimate"] - v) <= 1e-12
        assert abs(R.compute(m, counts)["estimate"] - v) <= 1e-12


SYMMETRIC = ["fleiss_kappa", "conger_kappa", "light_kappa", "gwet_ac1", "gwet_ac2", "brennan_prediger", "kripp_nominal",
             "kripp_ordinal", "kripp_interval", "kripp_ratio", "percent_agreement", "icc_1_1", "icc_2_1", "icc_3_k", "kendall_w"]


@pytest.mark.parametrize("method", SYMMETRIC)
def test_rater_order_invariance(method):
    e = next(x for x in REG["examples"] if x["method"] == method); data = load(e)
    if data.layout == "counts":
        pytest.skip("counts carry no rater order")
    rev = R.RaterData("wide", data.table.iloc[:, ::-1].copy())
    assert abs(R.compute(method, data)["estimate"] - R.compute(method, rev)["estimate"]) <= 1e-12


@pytest.mark.parametrize("method", ["cohen_kappa", "scott_pi", "bangdiwala_b", "conger_kappa", "light_kappa", "gwet_ac1",
                                    "brennan_prediger", "kripp_nominal", "percent_agreement"])
def test_nominal_methods_are_invariant_to_category_relabeling(method):
    e = next(x for x in REG["examples"] if x["method"] == method); data = load(e)
    cats = data.categories(); relabel = {c: f"K{len(cats) - i}" for i, c in enumerate(cats)}
    other = R.RaterData("wide", data.table.apply(lambda col: col.map(lambda v: relabel.get(v, v))).astype(object))
    assert abs(R.compute(method, data)["estimate"] - R.compute(method, other)["estimate"]) <= 1e-12


def test_weighted_kappa_is_sensitive_to_category_order():
    e = next(x for x in REG["examples"] if x["method"] == "cohen_kappa_quadratic"); data = load(e)
    perm = {1: 3, 2: 1, 3: 5, 4: 2, 5: 4}
    other = R.RaterData("wide", data.table.apply(lambda c: c.map(lambda v: float(perm[int(v)]))).astype(object))
    assert abs(R.compute("cohen_kappa_quadratic", data)["estimate"] - R.compute("cohen_kappa_quadratic", other)["estimate"]) > 1e-3


def _frame(**cols):
    return pd.DataFrame({"Birim_ID": [f"B{i}" for i in range(len(next(iter(cols.values()))))], **cols})


@pytest.mark.parametrize("case", ["duplicate_id", "mixed", "text_number", "whitespace", "case", "bool", "inf"])
def test_coding_problems_are_reported_never_repaired(case):
    a, b = [1, 2, 3, 1, 2], [1, 2, 2, 1, 3]
    df = _frame(R1=a, R2=b).astype(object)
    if case == "duplicate_id": df.loc[1, "Birim_ID"] = "B0"
    if case == "mixed": df.loc[0, "R1"] = "A"
    if case == "text_number": df.loc[0, "R2"] = "2"
    if case == "whitespace": df["R1"] = ["Evet", "Hayır", "Evet ", "Evet", "Hayır"]; df["R2"] = ["Evet"] * 5
    if case == "case": df["R1"] = ["Evet", "evet", "Hayır", "Evet", "Hayır"]; df["R2"] = ["Evet"] * 5
    if case == "bool": df.loc[0, "R1"] = True
    if case == "inf": df["R1"] = df["R1"].astype(float); df.loc[0, "R1"] = np.inf
    data = R.from_wide(df, "Birim_ID", ["R1", "R2"])
    assert any(sev == "error" for sev, _ in data.issues)
    assert not R.applicable("cohen_kappa", data)[0]
    with pytest.raises(R.RaterDataError):
        R.compute("cohen_kappa", data)


def test_teaching_invalid_example_is_detected():
    df = pd.read_csv(EX / "invalid_teaching.csv", dtype=object).apply(lambda c: c if c.name == "Birim_ID" else c.map(
        lambda v: v if pd.isna(v) or v == "2" else float(v)))
    data = R.from_wide(df, "Birim_ID", list(df.columns[1:]))
    msgs = " ".join(m for _, m in data.issues)
    assert "Tekrarlanan" in msgs and "metin olarak" in msgs and data.has_missing


def test_structural_fail_closed_rules():
    with pytest.raises(R.RaterDataError, match="ortalanmaz"):
        R.from_long(pd.DataFrame({"u": ["a", "a", "b"], "r": ["x", "x", "x"], "s": [1, 2, 1]}), "u", "r", "s")
    with pytest.raises(R.RaterDataError, match="aynı sayıda"):
        R.from_counts(pd.DataFrame({"id": ["a", "b"], "k1": [2, 3], "k2": [1, 1]}), "id", ["k1", "k2"])
    miss = R.from_wide(_frame(R1=[1, 2, np.nan, 1], R2=[1, 2, 2, 1], R3=[2, 2, 2, 1]), "Birim_ID", ["R1", "R2", "R3"])
    assert not R.applicable("fleiss_kappa", miss)[0] and R.applicable("kripp_nominal", miss)[0]
    four = load(next(x for x in REG["examples"] if x["method"] == "conger_kappa"))
    assert not R.applicable("cohen_kappa", four)[0] and not R.applicable("mcnemar", four)[0]
    ordinal = load(next(x for x in REG["examples"] if x["method"] == "stuart_maxwell"))
    assert not R.applicable("pabak", ordinal)[0]
    nominal = load(next(x for x in REG["examples"] if x["method"] == "cohen_kappa"))
    assert not R.applicable("cohen_kappa_linear", nominal)[0] and not R.applicable("pearson", nominal)[0]
    single = R.from_wide(_frame(R1=[1, 2, 2, 3], R2=[1, 2, 3, 3]), "Birim_ID", ["R1", "R2"])
    with pytest.raises(R.RaterDataError, match="tekil"):
        R.compute("stuart_maxwell", single)
    const = R.from_wide(_frame(R1=[1, 1, 1], R2=[1, 1, 1]), "Birim_ID", ["R1", "R2"])
    out = R.run("cohen_kappa", const)
    assert not np.isfinite(out["result"]["estimate"]) and "tanımsız" in out["interpretation"]


def test_unvalidated_outputs_are_hidden_and_labels_never_certified():
    e = next(x for x in REG["examples"] if x["method"] == "icc_2_k")
    out = R.run("icc_2_k", load(e))
    assert "ci95" in out["result"] and "ci95" not in out["shown"] and "kısmi" in out["status"]
    fleiss = R.run("fleiss_kappa", load(next(x for x in REG["examples"] if x["method"] == "fleiss_kappa")))
    assert set(fleiss["shown"]) == {"estimate"}
    assert not any("CERTIFIED" in v["label"] and "NOT" not in v["label"] for v in EVID.values())


def test_independent_r_references(tmp_path):
    rs, irr, cac = shutil.which("Rscript"), os.getenv("IRR_SRC"), os.getenv("IRRCAC_SRC")
    if not (rs and irr and cac):
        if os.getenv("REQUIRE_RATER_R") == "1": pytest.fail("R rater references required")
        pytest.skip("NOT EXECUTED: Rscript/IRR_SRC/IRRCAC_SRC unavailable")
    out = tmp_path / "reference.csv"
    run = subprocess.run([rs, str(ROOT / "validation/rater_reliability/reference.R"), irr, cac, str(EX), str(out)],
                         capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    df, status, _ = compare(out)
    df.to_csv(tmp_path / "comparison.csv", index=False)
    assert len(df) == json.loads((ROOT / "validation/rater_reliability/evidence.json").read_text())["n_comparisons"]
    assert {m: s["label"] for m, s in status.items()} == {m: s["label"] for m, s in EVID.items()}
    assert {m: s["failed_outputs"] for m, s in status.items()} == {m: s["failed_outputs"] for m, s in EVID.items()}


def test_page_example_workflow_and_exports():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string("from ui.rater_reliability_view import render\nrender()", default_timeout=120)
    at.run(); assert not at.exception
    assert any("Veri Setimi Nasıl Düzenlemeliyim?" in e.label for e in at.expander)
    at.selectbox(key="rr_example_choice").set_value("bland_altman_paired").run()
    at.button(key="rr_load_example").click().run(); assert not at.exception
    at.button(key="rr_run").click().run(); assert not at.exception
    saved = at.session_state["rr_results"]["out"]
    assert list(saved) == ["bland_altman"] and saved["bland_altman"]["shown"]["bias"] < 0
    from ui.rater_reliability_view import _exports
    e = next(x for x in REG["examples"] if x["method"] == "bland_altman")
    files = _exports(saved, load(e), e["config"])
    assert {"bulgular.csv", "bulgular.xlsx", "rapor.docx", "bland_altman.png", "analiz_yapilandirmasi.json",
            "puanlayici_guvenirligi.zip"} <= set(files)
    names = zipfile.ZipFile(io.BytesIO(files["puanlayici_guvenirligi.zip"])).namelist()
    assert "rapor.docx" in names and "bland_altman.png" in names
    from docx import Document
    text = "\n".join(p.text for p in Document(io.BytesIO(files["rapor.docx"])).paragraphs)
    assert "Bland–Altman" in text and "Sürümler" in text and "sertifika değildir" in text
    at.selectbox(key="rr_example_choice").set_value("cohen_kappa_nominal").run()
    at.button(key="rr_load_example").click().run()
    assert "rr_results" not in at.session_state                     # stale results cleared on new data


def test_platform_example_dataset_can_be_analysed_on_the_page():
    from streamlit.testing.v1 import AppTest
    import core.example_datasets as E
    d = next(x for x in E.available(E.RATER_PAGE) if x.level == 0); b = E.materialize(d.id)
    at = AppTest.from_string("from ui.rater_reliability_view import render\nrender()", default_timeout=120)
    at.session_state["raw_df"] = b["data"]; at.session_state["item_cols"] = b["items"]; at.session_state["id_col"] = b["id"]
    at.run(); at.button(key="rr_use_platform").click().run(); assert not at.exception
    at.multiselect(key="rr_sel_kategorik").set_value(["cohen_kappa_quadratic"]).run()
    at.button(key="rr_run").click().run(); assert not at.exception
    out = at.session_state["rr_results"]["out"]["cohen_kappa_quadratic"]
    assert np.isfinite(out["result"]["estimate"]) and out["n_raters"] == 2 and out["n_units"] == d.n
