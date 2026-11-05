"""One-click 1PL–4PL ML comparison table and a deterministic Turkish interpretation.

All four models are Python mirt marginal ML (EM) fits on the same complete 0/1 matrix with the
same N(0,1) latent distribution and the same Gauss–Hermite grid; MAP objectives are never mixed in.
AIC = -2LL + 2k and BIC = -2LL + k ln(N), N = number of persons (independent response vectors),
k = freely estimated parameters reported by the library; both are recomputed and cross-checked.
Eligibility for selection: converged, finite, N and AIC/BIC consistent, and identification not ruled
out by the platform policy (4PL ML has no supported identification evidence, so it is never selected).
"""
import math

import numpy as np
import pandas as pd

MODELS = ("1PL", "2PL", "3PL", "4PL")
SOLVER = dict(max_iter=2000, tol=1e-6, n_quadpts=41)          # identical for all four fits
BOUNDS = {"discrimination": (.1, 5.), "difficulty": (-6., 6.), "guessing": (0., .5), "upper": (.5, 1.)}
DESCRIPTION = {"1PL": "tüm maddelerde ortak (a = 1) ayırt edicilik", "2PL": "maddeye göre değişen ayırt edicilik",
               "3PL": "ek olarak alt asimptot (şans) parametresi", "4PL": "alt ve üst asimptot parametreleri"}
_DASH = "—"
# Evidence-based 4PL ML gate (not an unconditional exclusion): selectable only when a certification
# record supports it. Current evidence (validation/FOUR_PL_ML_DECISION.md, four_pl_gh_refinement/result.json,
# evidence/p1_3pl4pl, evidence/p7_binary34_se): weak practical identification, many boundary estimates,
# GH101 solution not stationary under the GH161 audit, R EM nonconvergence, failed CI coverage.
FOUR_PL_ML_CERTIFIED_SCOPE = None


def four_pl_gate(n_persons, n_items, boundary):
    scope = FOUR_PL_ML_CERTIFIED_SCOPE
    if scope and n_persons >= scope["min_n"] and n_items >= scope["min_items"] and boundary == 0:
        return True, ""
    return False, ("4PL ML seçime alınmadı: tanımlanabilirlik, kuadratür durağanlığı ve GA kapsaması "
                   "kanıtlanmadı (FOUR_PL_ML_DECISION.md)" + (f"; {boundary} sınır kestirimi" if boundary else ""))


def boundary_count(fit, tol=1e-4):
    try:
        co = fit.coef()
    except Exception:
        return 0
    count = 0
    for column, (lo, hi) in BOUNDS.items():
        if column in co:
            v = pd.to_numeric(co[column], errors="coerce").to_numpy(float)
            count += int(np.sum((np.abs(v - lo) <= tol) | (np.abs(v - hi) <= tol)))
    return count


def summarize_fit(model, fit, n_persons, n_items=None):
    """Row dict for one model; never raises for an unusable fit."""
    row = dict(Model=model, k=None, LogLik=None, AIC=None, BIC=None, converged=False, boundary=0,
               eligible=False, reason="")
    if fit is None:
        row["reason"] = "kestirim başarısız"
        return row
    try:
        ll, k = float(fit.log_likelihood), int(fit.n_parameters)
        n_obs, converged = int(fit.n_observations), bool(fit.converged)
    except Exception:
        row["reason"] = "kestirim çıktısı okunamadı"
        return row
    row.update(converged=converged, boundary=boundary_count(fit))
    if n_items is None:
        try:
            n_items = len(fit.coef())
        except Exception:
            n_items = 0
    if not math.isfinite(ll) or k < 1:
        row["reason"] = "sonlu olmayan logLik"
        return row
    aic, bic = -2 * ll + 2 * k, -2 * ll + k * math.log(n_persons)
    row.update(k=k, LogLik=ll, AIC=aic, BIC=bic)
    if n_obs != n_persons:
        row["reason"] = "Karşılaştırılamaz: farklı gözlem sayısı"
    elif abs(aic - float(fit.aic)) > 1e-6 * max(1, abs(aic)) or abs(bic - float(fit.bic)) > 1e-6 * max(1, abs(bic)):
        row["reason"] = "Karşılaştırılamaz: kütüphane AIC/BIC tanımı farklı"
    elif not converged:
        row["reason"] = "yakınsamadı"
    elif model == "4PL" and not four_pl_gate(n_persons, n_items, row["boundary"])[0]:
        row["reason"] = four_pl_gate(n_persons, n_items, row["boundary"])[1]
    else:
        row.update(eligible=True, reason="uygun")
    return row


def comparison_table(rows):
    """Exactly four rows in fixed order with ΔAIC/ΔBIC relative to eligible models only."""
    by = {r["Model"]: r for r in rows}
    ordered = [by.get(m, dict(Model=m, k=None, LogLik=None, AIC=None, BIC=None, converged=False, boundary=0,
                             eligible=False, reason="çalıştırılmadı")) for m in MODELS]
    eligible = [r for r in ordered if r["eligible"]]
    min_aic = min((r["AIC"] for r in eligible), default=None)
    min_bic = min((r["BIC"] for r in eligible), default=None)
    out = []
    for r in ordered:
        fmt = lambda v, d=3: _DASH if v is None else f"{v:.{d}f}"
        status = "Uygun" if r["eligible"] else r["reason"][:1].upper() + r["reason"][1:]
        if r["eligible"] and r["boundary"]:
            status += f" ({r['boundary']} sınır kestirimi)"
        out.append({"Model": r["Model"], "k": _DASH if r["k"] is None else str(r["k"]), "LogLik": fmt(r["LogLik"]),
                    "AIC": fmt(r["AIC"]), "BIC": fmt(r["BIC"]),
                    "ΔAIC": fmt(r["AIC"] - min_aic) if r["eligible"] else _DASH,
                    "ΔBIC": fmt(r["BIC"] - min_bic) if r["eligible"] else _DASH,
                    "Yakınsama": "Evet" if r["converged"] else ("Hayır" if r["k"] is not None else _DASH),
                    "Durum": status})
    return pd.DataFrame(out, columns=["Model", "k", "LogLik", "AIC", "BIC", "ΔAIC", "ΔBIC", "Yakınsama", "Durum"])


def _evidence(delta):
    if delta < 2:
        return "zayıf"
    if delta < 6:
        return "olumlu"
    if delta < 10:
        return "güçlü"
    return "çok güçlü"


def interpretation(rows):
    """One Turkish paragraph built only from the computed rows."""
    by = {r["Model"]: r for r in rows}
    eligible = [by[m] for m in MODELS if m in by and by[m]["eligible"]]
    excluded = [f"{m} ({by[m]['reason']})" for m in MODELS if m in by and not by[m]["eligible"]]
    definitions = ("Modeller iç içe yapıdadır: 1PL " + DESCRIPTION["1PL"] + ", 2PL " + DESCRIPTION["2PL"] + ", 3PL "
                   + DESCRIPTION["3PL"] + " ve 4PL " + DESCRIPTION["4PL"] + " içerir.")
    tail = (" Karşılaştırma dışı bırakılan modeller: " + "; ".join(excluded) + "; bu modeller bilgi ölçütü değeri "
            "küçük olsa bile önerilmez." if excluded else "")
    if not eligible:
        return (definitions + " Aynı veri ve aynı marjinal olabilirlik tanımıyla geçerli biçimde kestirilmiş model "
                "bulunmadığından savunulabilir bir tercih edilen model seçilemez." + tail)
    best = min(eligible, key=lambda r: r["BIC"])
    best_aic = min(eligible, key=lambda r: r["AIC"])
    others = sorted((r for r in eligible if r is not best), key=lambda r: r["BIC"])
    text = [definitions, f"Seçime uygun {len(eligible)} model arasında en küçük BIC değeri {best['Model']} modeline "
            f"aittir (BIC = {best['BIC']:.2f}, k = {best['k']})."]
    if best_aic is best:
        text.append("AIC de aynı modeli işaret etmektedir; iki ölçüt uyumludur.")
    else:
        text.append(f"AIC ise {best_aic['Model']} modelini işaret etmektedir; BIC örneklem büyüklüğüyle artan daha "
                    "güçlü bir karmaşıklık cezası uyguladığı için daha az parametreli çözümü yeğlemiştir, AIC ise "
                    "tahmin doğruluğuna daha fazla ağırlık verir.")
    if others:
        nxt = others[0]; d = nxt["BIC"] - best["BIC"]
        richer = nxt["k"] > best["k"]
        text.append(f"En yakın rakip {nxt['Model']} için ΔBIC = {d:.2f}; Raftery'nin (1995) sezgisel sınırlarına göre bu "
                    f"fark {_evidence(d)} kanıta karşılık gelir, ancak bu sınırlar kesin kanıt değildir.")
        text.append(("Ek parametrelerin sağladığı olabilirlik artışı, artan karmaşıklığı karşılamamaktadır."
                     if richer else
                     f"{best['Model']} modelinin ek parametreleri, olabilirlikte karmaşıklık cezasını aşan bir iyileşme "
                     "sağlamaktadır."))
    if best["boundary"]:
        text.append(f"Seçilen modelde {best['boundary']} parametre kestirim sınırındadır; bu durumda olabilirlik oranı "
                    "testleri düzenli koşulları sağlamaz ve ilgili parametreler dikkatle yorumlanmalıdır.")
    return " ".join(text) + tail
