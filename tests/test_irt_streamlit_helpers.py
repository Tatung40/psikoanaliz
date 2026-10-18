"""B7: Streamlit IRT screen helpers (ui/irt_optimization.py) and the EM/KKT stage split."""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import core.irt as irt
from core.irt import fit_irt, probabilities, estimate_ability
from core.irt_hessian import observed_information_se
from core.irt_model_reliability import model_reliability
from core.irt_quadrature_validation import marginal_loglik_at_fixed_parameters
from core.irt_simulation import simulate_irt
from ui import irt_optimization as ui

FIX = Path(__file__).parent / "fixtures" / "mirt"


@lru_cache(maxsize=None)
def _small():
    truth = pd.DataFrame({"Madde": [f"M{i+1}" for i in range(5)], "a": np.linspace(1.2, 2., 5),
                          "b": np.linspace(-1., 1., 5), "c": np.linspace(.12, .2, 5), "d": 1.})
    return simulate_irt(600, truth, seed=12)[0]


@lru_cache(maxsize=None)
def _k8():
    return pd.read_csv(FIX / "data_4pl_free_k8.csv")


@lru_cache(maxsize=None)
def _refined_4pl():
    """EM stops at 70 cycles without logL convergence; refinement meets KKT (B6)."""
    s = ui.calibration_settings("4PL", 21, 70, starts=1, refine=True)
    return ui.run_irt_calibration(_k8(), s)


def test_settings_for_1pl_2pl_never_carry_refinement():
    for model in ("1PL", "2PL"):
        s = ui.calibration_settings(model, 21, 70, a_upper=10., d_lower=.2, starts=3, refine=True)
        assert not ui.refine_available(model)
        assert (s["refine"], s["starts"], s["a_upper"], s["d_lower"]) == (False, 1, 4.0, .65)
    assert ui.refine_available("3PL") and ui.refine_available("4PL")
    assert ui.calibration_settings("3PL", 21, 70, d_lower=.2)["d_lower"] == .65


def test_default_refine_false_matches_plain_multistart():
    from core.irt_multistart import fit_irt_multistart
    s = ui.calibration_settings("3PL", 15, 70, starts=2)
    r = ui.run_irt_calibration(_small(), s)
    plain = fit_irt_multistart(_small(), "3PL", starts=(0., -1.), points=15, max_iter=70)["fit"]
    pd.testing.assert_frame_equal(r["items"], plain["items"])
    assert r["refinement"] == {"requested": False}
    assert r["settings"]["method"] == "EM" and r["settings"]["seconds"] >= 0
    assert set(r["multistart_audit"]["İyileştirme"]) == {"uygulanmadı"}


def test_refine_true_reaches_every_start(monkeypatch):
    seen = []
    real = irt.fit_irt
    def spy(*args, **kwargs):
        seen.append((kwargs.get("refine"), kwargs.get("points"), kwargs.get("max_iter"),
                     kwargs.get("a_upper"), kwargs.get("d_lower")))
        return real(*args, **kwargs)
    import core.irt_multistart as ms
    monkeypatch.setattr(ms, "fit_irt", spy)
    s = ui.calibration_settings("4PL", 15, 30, a_upper=6., d_lower=.4, starts=3, refine=True)
    r = ui.run_irt_calibration(_small(), s)
    assert seen == [(True, 15, 30, 6., .4)] * 3
    assert len(r["multistart_audit"]) == 3
    assert set(r["multistart_audit"]["İyileştirme"]) <= {"kabul edildi", "reddedildi"}
    assert r["settings"]["method"] == "EM + L-BFGS-B"


def test_1pl_2pl_path_has_no_multistart_or_refinement():
    r = ui.run_irt_calibration(_small(), ui.calibration_settings("2PL", 15, 30, refine=True))
    assert "multistart_audit" not in r and r["refinement"] == {"requested": False}
    msgs = ui.convergence_messages(r)
    assert all(level == "warning" and "EM yakınsama" in text for level, text in msgs)
    assert not any("L-BFGS-B" in text or "KKT" in text for _, text in msgs)


def test_em_and_kkt_stages_are_reported_separately():
    r = _refined_4pl()
    conv, rf = r["convergence"], r["refinement"]
    assert r["iterations"] >= 70 and not conv["loglik_converged"] and not rf["em_converged"]
    assert conv["em_status"] == "not_converged"
    assert r["converged"] and conv["status"] == "converged" and conv["criterion"] == "kkt_after_refinement"
    msgs = ui.convergence_messages(r)
    texts = " ".join(t for _, t in msgs)
    assert "sonuçlar geçicidir" not in texts                        # no misleading EM-only warning
    assert "Aşama 1 (EM): logL ölçütü" in texts and "sağlanmadı" in texts
    assert "Aşama 2 (L-BFGS-B): iyileştirme kabul edildi" in texts
    assert ui.KKT_CAVEAT in texts
    assert any(level == "warning" and "kutu sınırlarına" in t for level, t in msgs)   # 4PL boundary optimum
    table = dict(zip(*ui.optimization_summary(r).T.values))
    assert table["Kestirim yöntemi"] == "EM + L-BFGS-B" and table["İyileştirme"] == "Kabul edildi"
    assert table["EM aşaması"] == "Yakınsamadı" and table["Nihai yakınsama durumu"] == "Yakınsadı"
    assert table["Kullanılan yakınsama ölçütü"] == "İyileştirme sonrası KKT"
    assert table["Gauss–Hermite düğümü"] == "21"
    rel = model_reliability(r)
    assert rel["Kestirim yöntemi"] == "EM + L-BFGS-B" and rel["Yakınsama ölçütü"] == "kkt_after_refinement"
    assert any("EM aşaması logL ölçütüyle yakınsamadı" in a for a in rel["Açıklamalar"])


def test_rejected_refinement_shows_reason_and_em_result(monkeypatch):
    em = fit_irt(_small(), "3PL", points=15)
    monkeypatch.setattr(irt, "_refine_minimize", lambda f, x0, **k: SimpleNamespace(
        x=np.full_like(np.asarray(x0, float), np.nan), success=True, message="fake", nit=1))
    r = ui.run_irt_calibration(_small(), ui.calibration_settings("3PL", 15, 70, refine=True))
    pd.testing.assert_frame_equal(r["items"], em["items"])
    msgs = ui.convergence_messages(r)
    assert msgs[0][0] == "warning" and "reddedildi" in msgs[0][1] and "Sonlu olmayan" in msgs[0][1]
    assert "EM sonucudur" in msgs[0][1]
    table = dict(zip(*ui.optimization_summary(r).T.values))
    assert table["İyileştirme"] == "Reddedildi" and "Sonlu olmayan" in table["Ret nedeni"]
    assert set(r["multistart_audit"]["İyileştirme"]) == {"reddedildi"}
    rel = model_reliability(r)
    assert any("İyileştirme reddedildi" in a for a in rel["Açıklamalar"])


def test_plain_em_nonconvergence_keeps_existing_warning():
    r = ui.run_irt_calibration(_small(), ui.calibration_settings("3PL", 15, 2))
    assert not r["converged"]
    assert ui.convergence_messages(r)[0][1].startswith("EM yakınsama ölçütü sağlanmadı")


def test_reported_quantities_use_final_accepted_parameters():
    r = _refined_4pl()
    items = r["items"]
    a, b, c, d = (items[q].to_numpy(float) for q in "abcd")
    np.testing.assert_allclose(r["icc"], probabilities(r["grid"], a, b, c, d), atol=1e-12)
    assert r["loglik"] == pytest.approx(marginal_loglik_at_fixed_parameters(_k8(), items, points=21), abs=1e-8)
    assert r["loglik"] == pytest.approx(r["refinement"]["loglik_refined"], abs=1e-8)
    assert r["aic"] == pytest.approx(2 * r["param_count"] - 2 * r["loglik"])
    ability = estimate_ability(_k8(), items, method="EAP")
    assert np.isfinite(ability["Yetenek (theta)"]).all()
    # boundary estimates keep the SE refusal
    _, reason = observed_information_se(_k8(), items, "4PL", points=ui.result_points(r, 41))
    assert "sınır" in reason


def test_stored_quadrature_and_settings_mismatch():
    r = _refined_4pl()
    assert ui.result_points(r, 41) == 21                     # slider value is ignored
    assert ui.result_points({"model": "2PL"}, 31) == 31       # legacy results fall back
    same = ui.calibration_settings("4PL", 21, 70, starts=1, refine=True)
    assert ui.settings_mismatch(r, same) == []
    other = ui.calibration_settings("4PL", 31, 70, starts=1, refine=False)
    assert ui.settings_mismatch(r, other) == ["Gauss–Hermite düğüm sayısı", "Gelişmiş kestirim"]


def test_structural_error_propagates_for_storage_cleanup():
    with pytest.raises(ValueError, match="yapısal olarak tanımlanamaz"):
        ui.run_irt_calibration(_small().iloc[:, :4], ui.calibration_settings("4PL", 15, 10, refine=True))
