"""B6: opt-in EM + bounded L-BFGS-B refinement (fit_irt(refine=True)).

Accuracy is judged against the common bounded optimum found in B5 and against
the R mirt 3PL MLE (B3), not against speed. Safeguards are exercised by
replacing the refinement optimizer (core.irt._refine_minimize) only.
"""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import core.irt as irt
from core.irt import fit_irt
from core.irt_hessian import observed_information_se
from core.irt_multistart import fit_irt_multistart
from core.irt_simulation import simulate_irt

FIX = Path(__file__).parent / "fixtures" / "mirt"
BOX4 = {"a": (.15, 4.), "b": (-4., 4.), "c": (0., .35), "d": (.65, 1.)}


@lru_cache(maxsize=None)
def _small():
    truth = pd.DataFrame({"Madde": [f"M{i+1}" for i in range(5)], "a": np.linspace(1.2, 2., 5),
                          "b": np.linspace(-1., 1., 5), "c": np.linspace(.12, .2, 5), "d": 1.})
    return simulate_irt(600, truth, seed=12)[0]


@lru_cache(maxsize=None)
def _refit(case, model):
    return fit_irt(pd.read_csv(FIX / f"data_{case}.csv"), model, points=21, refine=True)


def test_default_path_is_unchanged():
    plain = fit_irt(_small(), "3PL", points=15)
    explicit = fit_irt(_small(), "3PL", points=15, refine=False)
    assert plain["refinement"] == {"requested": False}
    assert plain["convergence"]["criterion"] == "loglik"
    pd.testing.assert_frame_equal(plain["items"], explicit["items"])
    assert plain["loglik"] == explicit["loglik"]


@pytest.mark.parametrize("case", ["4pl_free_k5", "4pl_free_k8"])
def test_refined_4pl_reaches_b5_bounded_optimum(case):
    fit = _refit(case, "4PL")
    ref = pd.read_csv(FIX / "python_free4pl_results.csv")
    ref = ref[(ref.case == case) & (ref.source == "polished_from_mirt")]
    opt = ref.pivot(index="item", columns="param", values="value").loc[fit["items"]["Madde"]]
    rf = fit["refinement"]
    assert rf["accepted"] and rf["kkt_met"] and rf["max_kkt_final"] <= 1e-6
    assert rf["loglik_gain"] > 0 and rf["max_kkt_final"] < rf["max_kkt_em"]
    # stored with 12 significant digits (~1e-7 at |logL| ~1e4)
    assert fit["loglik"] >= ref.loglik.iloc[0] - 1e-6
    assert np.abs(fit["items"][list("abcd")].to_numpy() - opt[list("abcd")].to_numpy()).max() <= 2e-3
    assert fit["converged"] and fit["convergence"]["status"] == "converged"
    assert fit["convergence"]["criterion"] == "kkt_after_refinement"
    for q, (lo, hi) in BOX4.items():
        assert fit["items"][q].between(lo, hi).all()
    # boundary estimates: still no SE, and the boundary warning remains
    _, reason = observed_information_se(pd.read_csv(FIX / f"data_{case}.csv"), fit["items"], "4PL", points=21)
    assert "sınır" in reason
    assert any("optimizasyon sınırında" in w for w in fit["warnings"])


def test_refined_3pl_matches_mirt_mle_without_regression():
    fit = _refit("3pl_est", "3PL")
    r = pd.read_csv(FIX / "mirt_results.csv")
    e = r[r.case == "3pl_est"]
    cf = e[e.quantity == "fit_irtpars"].pivot(index="row", columns="col", values="value").loc[fit["items"]["Madde"]]
    mirt = np.column_stack([cf.a, cf.b, cf.g])
    assert np.abs(fit["items"][["a", "b", "c"]].to_numpy() - mirt).max() <= 1e-4
    assert fit["loglik"] >= float(e[e.quantity == "fit_loglik"].value.iloc[0]) - 1e-6
    assert fit["refinement"]["accepted"] and fit["warnings"] == []
    _, reason = observed_information_se(pd.read_csv(FIX / "data_3pl_est.csv"), fit["items"], "3PL", points=21)
    assert reason == ""


def _fake(make_x, success=True):
    def fake(fun, x0, **kwargs):
        return SimpleNamespace(x=make_x(np.asarray(x0, float)), success=success, message="fake", nit=1)
    return fake


@pytest.mark.parametrize("make_x,reason", [
    (lambda x0: x0 + np.where(np.arange(len(x0)) % 3 == 1, 1.5, 0.), "düşürdü"),   # shift b: worse logL
    (lambda x0: np.full_like(x0, np.nan), "Sonlu olmayan"),
    (lambda x0: np.where(np.arange(len(x0)) % 3 == 0, 10., x0), "sınırlarının dışına"),
])
def test_rejected_refinement_keeps_em_solution(monkeypatch, make_x, reason):
    em = fit_irt(_small(), "3PL", points=15)
    monkeypatch.setattr(irt, "_refine_minimize", _fake(make_x))   # success=True is not trusted
    fit = fit_irt(_small(), "3PL", points=15, refine=True)
    rf = fit["refinement"]
    assert not rf["accepted"] and reason in rf["reason"]
    pd.testing.assert_frame_equal(fit["items"], em["items"])
    assert fit["converged"] == em["converged"]
    assert any("reddedildi" in w and "EM çözümü korundu" in w for w in fit["warnings"])


def test_optimizer_exception_is_reported_not_hidden(monkeypatch):
    def boom(*args, **kwargs):
        raise FloatingPointError("overflow")
    monkeypatch.setattr(irt, "_refine_minimize", boom)
    fit = fit_irt(_small(), "3PL", points=15, refine=True)
    assert not fit["refinement"]["accepted"]
    assert "FloatingPointError" in fit["refinement"]["reason"]
    assert any("reddedildi" in w for w in fit["warnings"])


def test_unsuccessful_status_with_better_loglik_is_accepted_and_recorded(monkeypatch):
    real = irt._refine_minimize
    def flagged(fun, x0, **kwargs):
        out = real(fun, x0, **kwargs)
        return SimpleNamespace(x=out.x, success=False, message="ABNORMAL (simulated)", nit=out.nit)
    monkeypatch.setattr(irt, "_refine_minimize", flagged)
    fit = fit_irt(_small(), "3PL", points=15, refine=True)
    rf = fit["refinement"]
    assert rf["accepted"] and rf["optimizer_success"] is False and rf["loglik_gain"] >= 0
    assert rf["optimizer_message"] == "ABNORMAL (simulated)"


def test_refinement_restrictions_and_structural_check_kept():
    with pytest.raises(ValueError, match="3PL/4PL"):
        fit_irt(_small(), "2PL", refine=True)
    with pytest.raises(ValueError, match="yapısal olarak tanımlanamaz"):
        fit_irt(_small().iloc[:, :4], "4PL", refine=True)
    with pytest.raises(ValueError, match="İyileştirme ayarları"):
        fit_irt(_small(), "3PL", refine=True, kkt_tolerance=0.)


def test_multistart_forwards_refinement():
    out = fit_irt_multistart(_small(), "3PL", starts=(0., 1.), points=15, refine=True)
    assert out["fit"]["refinement"]["requested"] is True
