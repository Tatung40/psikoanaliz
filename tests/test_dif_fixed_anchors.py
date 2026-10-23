"""Anchor design gates and independent, previously pinned R likelihood benchmarks."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from core import irt_dif as D

F = Path(__file__).parent / "fixtures" / "irt_b10"


@pytest.fixture
def sample():
    x = pd.DataFrame(np.tile([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]], (100, 1)),
                     columns=list("ABCDE"))
    return x, ["R"] * 100 + ["F"] * 100


@pytest.fixture
def fit_calls(monkeypatch):
    calls = []

    class Fitter:
        def __init__(self, *args):
            self.model = args[3]
            self.cache = {}

        def __call__(self, types, information=False):
            calls.append(tuple(types))
            design = D.Design(self.model, types, 2)
            params = np.zeros(design.size)
            for j, (kind, _, _) in enumerate(design.labels):
                if kind == "a":
                    params[j] = 1.
            return dict(design=design, params=params, converged=True,
                        loglik=-1000. + sum(t != "anchor" for t in types),
                        cov=np.eye(design.size), se_reason="", mu_F=0., sd_F=1.)

    monkeypatch.setattr(D, "_Fitter", Fitter)
    return calls


@pytest.mark.parametrize("anchors,kwargs,match", [
    ([], {}, "en az|En az"), (["A"], {}, "en az|En az"),
    (["A", "A"], {}, "benzersiz"), (["A", "Z"], {}, "veri maddelerinde"),
    ("AB", {}, "listesi"), (list("ABCDE"), {}, "test edilecek"),
    (["A", "B"], {"purify": True}, "arındırmayı kapatın"),
])
def test_invalid_fixed_design_fails_before_fitting(sample, fit_calls, anchors, kwargs, match):
    x, g = sample
    settings = dict(purify=False, min_anchors=2)
    settings.update(kwargs)
    with pytest.raises(ValueError, match=match):
        D.dif_analysis(x, g, "R", "F", fixed_anchors=anchors, **settings)
    assert not fit_calls


@pytest.mark.parametrize("minimum", [0, 1, 2.5, True])
def test_anchor_minimum_cannot_be_silently_truncated(sample, fit_calls, minimum):
    x, g = sample
    with pytest.raises(ValueError, match="tamsayı"):
        D.dif_analysis(x, g, "R", "F", min_anchors=minimum)
    assert not fit_calls


def test_fixed_anchors_remain_invariant_and_outside_test_family(sample, fit_calls):
    x, g = sample
    out = D.dif_analysis(x, g, "R", "F", fixed_anchors=["A", "B"], purify=False)
    t = out["table"].set_index("Madde")
    assert all(types[:2] == ("anchor", "anchor") for types in fit_calls)
    assert t.loc[["A", "B"], "p"].isna().all()
    assert t.loc[["A", "B"], "Karar"].eq("Sabit çapa (test edilmedi)").all()
    assert t.loc[list("CDE"), "Test çapaları"].eq("A, B").all()
    assert t.loc[list("CDE"), "Çapa sayısı (test)"].eq(2).all()
    assert out["anchor_policy"] == "fixed" and out["anchors"] == ["A", "B"]
    assert not any("bu veriyle seçildi" in w for w in out["warnings"])


@pytest.mark.parametrize("missing", [False, True])
def test_degenerate_items_are_never_fixed_anchors(sample, fit_calls, missing):
    x, g = sample
    x.loc[100:, "A"] = np.nan if missing else 1
    with pytest.raises(ValueError, match="her iki grupta"):
        D.dif_analysis(x, g, "R", "F", fixed_anchors=["A", "B"], purify=False)
    assert not fit_calls


def test_each_studied_item_needs_the_full_anchor_minimum(sample, fit_calls):
    x, g = sample
    with pytest.raises(ValueError, match="en az 5 geçerli çapa"):
        D.dif_analysis(x, g, "R", "F", purify=False, min_anchors=5)
    assert not fit_calls


@pytest.mark.parametrize("qsets,rounds,match", [
    ([{0}, {1}], 1, "kararlı"),
    ([{0}, {1}, {0}], 5, "kararlı"),
    ([{0, 1, 2, 3}], 5, "geçerli çapa"),
    ([{0}, {0}, {1}], 5, "Son kestirim"),
])
def test_unstable_or_depleted_purification_never_returns_decisions(sample, fit_calls, monkeypatch,
                                                                  qsets, rounds, match):
    answers = iter(qsets)
    # Advance one scripted BH family per round, independently of optimizer details.
    def bh(p):
        flagged = next(answers)
        return np.array([.001 if i in flagged else .9 for i in range(len(p))])
    monkeypatch.setattr(D, "benjamini_hochberg", bh)
    x, g = sample
    with pytest.raises(ValueError, match=match):
        D.dif_analysis(x, g, "R", "F", max_rounds=rounds)


@pytest.mark.parametrize("case,model", [("dif1pl_uniform", "1PL"), ("dif2pl_mixed", "2PL")])
def test_predeclared_design_matches_pinned_r_single_item_lr(case, model):
    data = pd.read_csv(F / f"data_{case}.csv")
    x = data.drop(columns="Grup")
    anchors = [c for c in x if c != "M3"]
    out = D.dif_analysis(x, data.Grup, "R", "F", model=model, fixed_anchors=anchors,
                         purify=False, quadrature="mirt")
    t = out["table"].set_index("Madde")
    refs = pd.read_csv(F / "r_b10_results.csv")
    r = refs[(refs.case == case) & (refs.row == "M3")]
    uniform = float(r.loc[r.quantity == "lr_uniform", "value"].iloc[0])
    tol = 1e-4 if model == "1PL" else 1e-5  # existing pinned EM reference tolerances
    if model == "1PL":
        assert t.loc["M3", "LR χ² (sd=1)"] == pytest.approx(uniform, abs=tol)
    else:
        omnibus = float(r.loc[r.quantity == "lr_omnibus", "value"].iloc[0])
        assert t.loc["M3", "LR χ² tek biçimli (sd=1)"] == pytest.approx(uniform, abs=tol)
        assert t.loc["M3", "LR χ² genel (sd=2)"] == pytest.approx(omnibus, abs=tol)
    assert out["joint"]["converged"] and out["joint"]["cov"] is not None
    assert t.loc[anchors, "p"].isna().all()
    assert t.loc["M3", "p"] == t.loc["M3", "q (BH)"]


def test_singular_information_blocks_fixed_anchor_inference(sample, fit_calls, monkeypatch):
    original = D._Fitter.__call__
    def singular(self, types, information=False):
        result = original(self, types, information)
        result["cov"] = None
        return result
    monkeypatch.setattr(D._Fitter, "__call__", singular)
    x, g = sample
    out = D.dif_analysis(x, g, "R", "F", fixed_anchors=["A", "B"], purify=False)
    tested = out["table"].iloc[2:]
    assert tested["p"].isna().all() and tested["q (BH)"].isna().all()
    assert tested["Karar"].eq("Çıkarım kullanılamıyor").all()
    assert tested["Not"].str.contains("bilgi matrisi").all()


def test_fixed_anchor_scale_and_joint_parameters_match_independent_r():
    case = "dif2pl_mixed"
    data = pd.read_csv(F / f"data_{case}.csv")
    x = data.drop(columns="Grup")
    anchors = pd.read_csv(F / f"py_dif_anchors_{case}.csv").anchor.tolist()
    out = D.dif_analysis(x, data.Grup, "R", "F", fixed_anchors=anchors, purify=False,
                         quadrature="mirt")
    refs = pd.read_csv(F / "r_b10_results.csv")
    refs = refs[refs.case == case]
    joint = out["joint"]
    for g, label in enumerate(["R", "F"]):
        ref = refs[refs.quantity == f"final_par_{label}"].pivot(index="row", columns="col", values="value")
        a, b = joint["design"].unpack(joint["params"], g)
        np.testing.assert_allclose(a, ref.loc[x.columns, "a1"], atol=.002, rtol=0)
        np.testing.assert_allclose(b, -ref.loc[x.columns, "d"] / ref.loc[x.columns, "a1"], atol=.002, rtol=0)
    pf = refs[refs.quantity == "final_par_F"]
    assert joint["mu_F"] == pytest.approx(float(pf.loc[pf.col == "MEAN_1", "value"].iloc[0]), abs=.002)
    assert joint["sd_F"] ** 2 == pytest.approx(float(pf.loc[pf.col == "COV_11", "value"].iloc[0]), abs=.003)
    assert joint["loglik"] >= float(refs.loc[refs.quantity == "final_loglik_fixed_sum", "value"].iloc[0]) - 1e-5
    t = out["table"].set_index("Madde")
    assert t.loc["M3", "DIF işareti"] and t.loc["M7", "DIF işareti"]
    assert t.loc["M3", "Δb (b_odak − b_ref)"] > .4
    assert t.loc["M7", "Δa (a_odak − a_ref)"] < -.5


def test_component_families_exclude_anchors_and_have_separate_bh(sample, fit_calls, monkeypatch):
    from scipy.stats import chi2
    original = D._Fitter.__call__
    uniform = [1., 1., .01, .04, .2]
    nonuniform = [1., 1., .02, .03, .5]
    def objective(self, types, information=False):
        fitted = original(self, types, information)
        gain = 0.
        for i, role in enumerate(types):
            if role != "anchor":
                gain += chi2.isf(uniform[i], 1) / 2
            if role == "ab":
                gain += chi2.isf(nonuniform[i], 1) / 2
        fitted["loglik"] = -1000. + gain
        return fitted
    monkeypatch.setattr(D._Fitter, "__call__", objective)
    x, g = sample
    out = D.dif_analysis(x, g, "R", "F", fixed_anchors=["A", "B"], purify=False)
    t = out["table"].set_index("Madde").loc[list("CDE")]
    np.testing.assert_allclose(t["q (BH, tek biçimli)"], [.03, .06, .2], atol=1e-12)
    np.testing.assert_allclose(t["q (BH, tek biçimli olmayan)"], [.045, .045, .5], atol=1e-12)
    np.testing.assert_allclose(t["Δb %95 GA üst"], 1.959963984540054 * np.sqrt(2))
    np.testing.assert_allclose(t["Δb %95 GA alt"], -t["Δb %95 GA üst"])


def test_missingness_and_group_filter_are_explicit(sample, fit_calls):
    x, g = sample
    x.loc[0, "A"] = np.nan
    x.loc[[100, 101], "A"] = np.nan
    x = pd.concat([x, pd.DataFrame([[np.nan] * 5, [0] * 5], columns=x.columns)], ignore_index=True)
    out = D.dif_analysis(x, g + ["R", "other"], "R", "F", fixed_anchors=["B", "C"], purify=False)
    missing = out["missingness"].set_index("Madde")
    assert missing.loc["A", "Eksik (referans)"] == 1 and missing.loc["A", "Eksik (odak)"] == 2
    assert out["excluded_rows"] == 2 and out["sizes"] == {"R": 100, "F": 100}
    assert out["kept_rows"].sum() == 200 and "MAR" in out["missing_policy"]


def test_degenerate_item_is_never_an_implicit_anchor(sample, fit_calls):
    x, g = sample
    x.loc[100:, "A"] = 1.
    out = D.dif_analysis(x, g, "R", "F", purify=False)
    assert "A" not in out["anchors"]
    assert all(types[0] == "ab" for types in fit_calls)
    assert out["table"].set_index("Madde").loc["A", "Karar"] == "Çıkarım kullanılamıyor"


def test_duplicate_identifiers_and_invalid_round_limit_fail_before_fit(sample, fit_calls):
    x, g = sample
    with pytest.raises(ValueError, match="pozitif tamsayı"):
        D.dif_analysis(x, g, "R", "F", max_rounds=0)
    x.columns = ["A", "A", "C", "D", "E"]
    with pytest.raises(ValueError, match="benzersiz"):
        D.dif_analysis(x, g, "R", "F")
    assert not fit_calls
