"""Testable helpers for the Streamlit IRT screen (B7). No Streamlit import.

They run the calibration with the user's settings, record those settings with
the result, and turn fit["refinement"] / fit["convergence"] / fit["warnings"]
into display rows and messages. No new statistics are computed here.
"""
from time import perf_counter
import numpy as np
import pandas as pd
from core.irt import fit_irt
from core.irt_multistart import fit_irt_multistart

REFINE_LABEL = "Gelişmiş kestirim (EM + L-BFGS-B)"
REFINE_HELP = ("EM kestiriminden sonra sınırlı sayısal optimizasyon uygulanır. Yakınsama "
               "kararlılığını iyileştirebilir; ancak parametre tanımlanabilirliğini garanti etmez.")
KKT_CAVEAT = ("KKT koşulunun sağlanması yerel, birinci mertebe bir koşuldur; global optimumu, "
              "parametre tanımlanabilirliğini veya model geçerliğini kanıtlamaz.")
SETTING_LABELS = {"points": "Gauss–Hermite düğüm sayısı", "max_iter": "Azami EM yinelemesi",
                  "a_upper": "a üst sınırı", "d_lower": "d alt sınırı",
                  "starts": "Başlangıç sayısı", "refine": "Gelişmiş kestirim"}


def refine_available(model):
    """The refinement option exists only for 3PL/4PL."""
    return model in ("3PL", "4PL")


def calibration_settings(model, points, max_iter, a_upper=4.0, d_lower=0.65, starts=1, refine=False):
    """Normalized settings; 1PL/2PL never carry refine/start/bound options."""
    if not refine_available(model):
        a_upper, d_lower, starts, refine = 4.0, 0.65, 1, False
    if model != "4PL":
        d_lower = 0.65
    return {"model": model, "points": int(points), "max_iter": int(max_iter),
            "a_upper": float(a_upper), "d_lower": float(d_lower), "starts": int(starts),
            "refine": bool(refine)}


def run_irt_calibration(data, settings):
    """Run the calibration exactly as configured and attach settings + timing."""
    s = dict(settings)
    model = s["model"]
    common = dict(points=s["points"], max_iter=s["max_iter"],
                  a_upper=s["a_upper"], d_lower=s["d_lower"])
    started = perf_counter()
    if refine_available(model):
        audit = fit_irt_multistart(data, model=model, starts=tuple([0., -1., 1.][:s["starts"]]),
                                   refine=s["refine"], **common)
        result = audit["fit"]
        result["start_logliks"] = audit["candidates"]["logLik"].dropna().tolist()
        result["start_convergences"] = audit["candidates"]["Yakınsama"].tolist()
        result["multistart_audit"] = audit["candidates"]
        result["multistart_summary"] = audit["summary"]
    else:
        result = fit_irt(data, model=model, **common)
    s["seconds"] = perf_counter() - started
    s["method"] = estimation_method(result)
    result["settings"] = s
    return result


def estimation_method(fit):
    return "EM + L-BFGS-B" if (fit.get("refinement") or {}).get("requested") else "EM"


def result_points(fit, fallback):
    """Quadrature used for the stored result (never the current slider value)."""
    return int((fit.get("settings") or {}).get("points", fallback))


def settings_mismatch(fit, current):
    """Labels of settings that differ between the stored result and the current UI."""
    stored = fit.get("settings") or {}
    return [SETTING_LABELS[k] for k in SETTING_LABELS
            if k in stored and k in current and stored[k] != current[k]]


def _fmt(value, digits=4):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "Evet" if value else "Hayır"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}g}" if abs(value) < 1e-3 or abs(value) >= 1e5 else f"{value:.{digits}f}"
    return str(value)


_STATUS = {"converged": "Yakınsadı", "loglik_only": "Yalnız logL ölçütü sağlandı",
           "not_converged": "Yakınsamadı"}
_CRITERION = {"loglik": "EM logL değişimi", "kkt_after_refinement": "İyileştirme sonrası KKT"}


def optimization_summary(fit):
    """Two-column table (Gösterge, Değer) built only from stored fit fields."""
    conv = fit.get("convergence") or {}
    rf = fit.get("refinement") or {}
    settings = fit.get("settings") or {}
    refined = bool(rf.get("requested"))
    em_status = conv.get("em_status", conv.get("status"))
    rows = [("Kestirim yöntemi", estimation_method(fit)),
            ("EM yinelemesi", fit.get("iterations")),
            ("EM aşaması", _STATUS.get(em_status, em_status))]
    if refined:
        rows += [("L-BFGS-B yinelemesi", rf.get("iterations")),
                 ("İyileştirme", "Kabul edildi" if rf.get("accepted") else "Reddedildi"),
                 ("Ret nedeni", rf.get("reason") or "—"),
                 ("logL (EM sonrası)", rf.get("loglik_em")),
                 ("logL (iyileştirme adayı)", rf.get("loglik_refined")),
                 ("logL kazanımı", rf.get("loglik_gain")),
                 ("KKT ihlali/kişi (EM sonrası)", rf.get("max_kkt_em")),
                 ("KKT ihlali/kişi (nihai)", rf.get("max_kkt_final")),
                 ("KKT eşiği", rf.get("kkt_tolerance")),
                 ("Optimize edici bildirimi", "başarılı" if rf.get("optimizer_success") else
                  f"başarısız ({rf.get('optimizer_message', '')})")]
    rows += [("Nihai yakınsama durumu", _STATUS.get(conv.get("status"), conv.get("status"))),
             ("Kullanılan yakınsama ölçütü", _CRITERION.get(conv.get("criterion", "loglik"))),
             ("Nihai logL", fit.get("loglik")),
             ("Sınırdaki parametre", conv.get("parameters_at_bound")),
             ("Gauss–Hermite düğümü", settings.get("points")),
             ("Hesaplama süresi (sn)", settings.get("seconds"))]
    return pd.DataFrame({"Gösterge": [r[0] for r in rows], "Değer": [_fmt(r[1]) for r in rows]})


def convergence_messages(fit):
    """Ordered (level, text) messages; keeps EM and KKT stages separate."""
    conv = fit.get("convergence") or {}
    rf = fit.get("refinement") or {}
    out = []
    em_ok = bool(conv.get("loglik_converged", fit.get("converged")))
    em_iter = fit.get("iterations")
    if not rf.get("requested"):
        if not fit.get("converged"):
            out.append(("warning", "EM yakınsama ölçütü sağlanmadı; sonuçlar geçicidir. "
                                   "Düğüm/yineleme sayısını artırın veya daha yalın model kullanın."))
        return out
    stage1 = ("Aşama 1 (EM): logL ölçütü sağlandı." if em_ok else
              f"Aşama 1 (EM): logL ölçütü {em_iter} yinelemede sağlanmadı.")
    if not rf.get("accepted"):
        out.append(("warning", f"{stage1} Aşama 2 (L-BFGS-B): iyileştirme reddedildi — "
                               f"{rf.get('reason', 'neden belirtilmedi')} Sunulan parametreler EM sonucudur."))
        if not fit.get("converged"):
            out.append(("warning", "EM sonucu yakınsamadığı için geçicidir; düğüm/yineleme sayısını "
                                   "artırın veya daha yalın model kullanın."))
        return out
    stage2 = (f"Aşama 2 (L-BFGS-B): iyileştirme kabul edildi (logL kazanımı "
              f"{_fmt(rf.get('loglik_gain'))}; KKT ihlali {_fmt(rf.get('max_kkt_final'))}/kişi, "
              f"eşik {_fmt(rf.get('kkt_tolerance'))}).")
    if rf.get("kkt_met"):
        out.append(("info", f"{stage1} {stage2} Sunulan parametreler bu aşamanın sonucudur; "
                            "yakınsama kararı KKT ölçütüne dayanır."))
    else:
        out.append(("warning", f"{stage1} {stage2} KKT eşiği sağlanmadı; çözüm durağan nokta olarak "
                               "doğrulanmadı ve geçici kabul edilmelidir."))
    out.append(("caption", KKT_CAVEAT))
    if conv.get("parameters_at_bound") and fit.get("model") == "4PL":
        out.append(("warning", "Çözüm kutu sınırlarına dayanan bir optimumdur: sınırdaki parametreler "
                               "programın izin verdiği sınır değerleridir, bilimsel kestirim değildir. "
                               "Bu optimum güvenilir bir 4PL kalibrasyonu olduğu anlamına gelmez; "
                               "daha yalın modelle karşılaştırın."))
    return out
