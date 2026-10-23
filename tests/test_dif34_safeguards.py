import numpy as np
import pandas as pd
import pytest
from core import irt_dif34 as D


def fit(**changes):
    return dict(dict(converged=True, regular=True, linked=True, bound_labels=[], loglik=-100.,
                     cov=np.eye(2), cov_conditional=False, quadrature_ok=True), **changes)


@pytest.mark.parametrize("change", [dict(cov=None), dict(cov=np.full((2, 2), np.nan)),
    dict(cov_conditional=True), dict(quadrature_ok=False)])
def test_information_and_integral_required_for_lr(change):
    stat, valid, note = D._lr(fit(**change), fit(loglik=-105.), 0)
    assert stat == 10 and not valid and "p değeri verilmedi" in note


def data():
    rng = np.random.default_rng(34)
    return pd.DataFrame(rng.binomial(1, .55, (240, 4)), columns=["A", "B", "C", "D"]), pd.Series(["R"] * 120 + ["F"] * 120)


@pytest.mark.parametrize("kwargs", [dict(model="4PL"), dict(min_anchors=True), dict(min_anchors=1),
    dict(max_rounds=0), dict(fixed_anchors=["A", "B"]), dict(purify=False, fixed_anchors=["A", "A"]),
    dict(purify=False, fixed_anchors=["UNKNOWN", "B"]), dict(purify=False, fixed_anchors=["A"])])
def test_invalid_settings_do_not_start_estimation(kwargs, monkeypatch):
    monkeypatch.setattr(D, "_Fitter34", lambda *a, **k: pytest.fail("Invalid input reached fitter"))
    x, g = data()
    with pytest.raises(ValueError): D.dif_analysis_34(x, g, "R", "F", **kwargs)


def test_constant_fixed_anchor_and_infinity_are_rejected():
    x, g = data(); x.A = 0
    with pytest.raises(ValueError, match="değişken"): D.dif_analysis_34(x, g, "R", "F", purify=False, fixed_anchors=["A", "B"])
    x = x.astype(float); x.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="Sonsuz"): D.dif_analysis_34(x, g, "R", "F")


def test_constant_items_cannot_supply_minimum_anchor_count(monkeypatch):
    x, g = data(); x.A = 0; x.B = 1
    # For C/D only one genuinely variable other item remains.
    class NoFit:
        def __init__(self, *args): pass
        def __call__(self, *args, **kwargs): pytest.fail("Invalid anchors reached estimation")
    monkeypatch.setattr(D, "_Fitter34", NoFit)
    with pytest.raises(ValueError, match="geçerli çapa"):
        D.dif_analysis_34(x, g, "R", "F", purify=False)


def test_cache_adds_information_and_audit_without_reoptimizing(monkeypatch):
    from core import irt_mg34 as M
    x, g = data(); values = x.to_numpy(float); obs = np.ones_like(values, bool)
    fitter = D._Fitter34(values, obs, (g == "F").to_numpy(int), "3PL", "gauss-hermite", 41, M.default_bounds())
    design = M.Design34("3PL", ["anchor"] * 4, 2)
    params = M.start_values(values, obs, design)
    cached = dict(params=params, design=design, converged=True, cov=None, info_done=False,
        at_bound=np.zeros(design.size, bool), loglik=-400.)
    fitter.cache[tuple(["anchor"] * 4)] = cached
    monkeypatch.setattr(D, "fit34", lambda *a, **k: pytest.fail("Repeated optimizer"))
    monkeypatch.setattr(D, "covariance", lambda *a, **k: (np.eye(design.size), "", 1.))
    result = fitter(["anchor"] * 4, information=True)
    assert result is cached and result["info_done"] and "quadrature_error" in result


def test_fixed_anchors_api_matches_existing_boxed_reference():
    from pathlib import Path
    root = Path(__file__).parent / "fixtures" / "irt_b11"
    # Existing R benchmark used these historically selected anchors; numerical
    # agreement is not evidence they were predeclared in the historical study.
    anchors = pd.read_csv(root / "py_dif_anchors_dif3pl.csv").anchor.tolist()
    frame = pd.read_csv(root / "data_dif3pl.csv")
    out = D.dif_analysis_34(frame.drop(columns="Grup"), frame.Grup, "R", "F", purify=False,
                            fixed_anchors=anchors, quadrature="mirt", points=41)
    assert out["anchors"] == anchors and out["anchor_policy"] == "predeclared fixed"
    table = out["table"].set_index("Madde")
    assert table.loc[anchors, "p"].isna().all()
    ref = pd.read_csv(root / "r_b11_boxed_wle_3PL.csv")
    expected = ref.loc[(ref.case == "dif3pl") & (ref.quantity == "box_final_loglik"), "value"].iloc[0]
    assert out["joint"]["loglik"] == pytest.approx(expected, abs=.005)


def test_unstable_purification_withholds_all_inference(monkeypatch):
    from core.irt_mg34 import Design34
    class Fitter:
        def __init__(self, *args): self.cache = {}
        def __call__(self, types, information=False):
            d = Design34("3PL", types, 2)
            p = np.zeros(d.size)
            for j, (kind, g, i) in enumerate(d.labels): p[j] = {"a": 1., "b": 0., "c": .15}[kind]
            ef = sum(types[i] != "anchor" for i in (4, 5))
            ll = -200 + (50 if ef == 1 else 51 if ef == 2 else 0) + sum(types[i] != "anchor" for i in range(4))
            f = dict(design=d, params=p, cov=np.eye(d.size), converged=True, regular=True, linked=True,
                loglik=ll, bound_labels=[], at_bound=np.zeros(d.size, bool), cov_conditional=False,
                quadrature_ok=True, se_reason="", mu_F=0., sd_F=1.)
            self.cache[tuple(types)] = f
            return f
    monkeypatch.setattr(D, "_Fitter34", Fitter)
    x = pd.DataFrame(np.random.default_rng(1).binomial(1, .5, (240, 6)), columns=list("ABCDEF"))
    out = D.dif_analysis_34(x, ["R"] * 120 + ["F"] * 120, "R", "F", max_rounds=1)
    assert not out["stable"] and out["table"].p.isna().all()
    assert not out["table"]["DIF işareti"].any()
    assert out["table"]["Not"].str.contains("kararsız").all()
    assert out["table"]["Δa %95 GA alt"].isna().all()
