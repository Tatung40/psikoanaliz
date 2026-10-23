"""Two-group DIF for the 3PL and 4PL models (B11).

Model: concurrent two-group bounded MML (core.irt_mg34) with P = c + (d - c) logistic(a(theta - b)),
reference theta ~ N(0, 1), focal mean/SD free, anchor items with all parameters equal.

Hypotheses (nested models for the studied item; all other items keep their current role):
    M0      item constrained equal across groups
    M_all   every free item parameter group-specific        -> omnibus LR, df = 3 (3PL) / 4 (4PL)
    M_p     only parameter p in {a, b, c[, d]} group-specific -> single-parameter LR, df = 1
M_p are the hypotheses in THIS parameterization: "a only" keeps b (not the slope-intercept d)
common. The tests are statements about parameters; the description of the DIF (direction,
crossing) is derived separately from the two group response functions, because with c > 0
(or d < 1) a difference in b alone is not a constant logit shift.

Inference policy (box-constrained ML; Self & Liang, 1987): an LR p-value is reported only when
M0 and the alternative converged (KKT), their finite log-likelihoods respect nestedness,
and no item or latent-distribution parameter is on an estimation bound in either model.
Boundary cases do not receive an ordinary chi-square p-value; the applicable nonstandard
reference distribution is not computed. Wald SEs come from
the conditional observed information and are reported only for interior studied parameters.
BH-FDR is applied to the available omnibus p-values. Items without a p-value cannot be flagged;
if their LR statistic exceeds the chi-square critical value they are removed from the anchor set
by a documented heuristic screen ("Çapa taraması"), which is not a DIF test. `inferential=False`
gives the descriptive analysis only (all-other-items anchors, no p-values).
p-values are conditional on the data-selected anchors; significant DIF is not proof of bias.
"""
import math
import numpy as np
import pandas as pd
from scipy.stats import chi2
from core.irt_mg import as_matrix
from core.irt_mg34 import Design34, PARAMS, default_bounds, fit as fit34, irf
from core.irt_dif import _prepare, _nested_lr
from core.rasch import _grid
from core.rasch_dif import benjamini_hochberg

SHAPES = {"against": "Odak grup aleyhine (eğriler odak dağılımında kesişmiyor)",
          "favour": "Odak grup lehine (eğriler odak dağılımında kesişmiyor)",
          "crossing": "Kesişimli (yön θ'ya göre değişiyor)", "none": "Fark yok denecek kadar küçük"}


class _Fitter34:
    def __init__(self, x, obs, groups, model, quadrature, points, bounds):
        self.args = (x, obs, groups); self.model = model; self.quadrature = quadrature
        self.points = points; self.bounds = bounds; self.cache = {}; self.base = None

    def __call__(self, types, information=False):
        key = tuple(types)
        got = self.cache.get(key)
        if got is not None and (got["cov"] is not None or not information or not got["converged"]
                                or got.get("info_done")):
            return got
        design = Design34(self.model, types, 2, self.bounds)
        start = got["params"] if got is not None else self._start(design)
        f = fit34(*self.args, design, self.quadrature, self.points, start=start, information=information)
        f["info_done"] = information
        self.cache[key] = f
        if self.base is None:
            self.base = f
        return f

    def _start(self, design):
        if self.base is None:
            return None
        bd = self.base["design"]; start = np.zeros(design.size)
        for j, (p, g, i) in enumerate(design.labels):
            start[j] = self.base["params"][bd.idx[p][max(g, 0), i]]
        start[design.latent:] = self.base["params"][bd.latent:]
        return start


def item_on_bound(f, i):
    return any(lab[2] == i for lab in f["bound_labels"])


def _lr(f_alt, f_null, i):
    """(statistic, usable for inference, note).

    Conservative rule: an asymptotic chi-square p-value is allowed only when both nested fits
    converged, are identified (anchored) and are fully regular, i.e. no free parameter of any
    item or of the latent distribution is boundary-active in either fit. This is a screening
    rule, not a proof that the chi-square reference holds; otherwise the LR is descriptive.
    """
    if not (f_alt["converged"] and f_null["converged"]):
        return np.nan, False, "Model yakınsamadı; p değeri verilmedi."
    stat, note = _nested_lr(f_alt["loglik"], f_null["loglik"])
    if note:
        return stat, False, note
    if not (f_alt.get("linked", True) and f_null.get("linked", True)):
        return stat, False, "Ortak ölçek tanımlanamıyor (çapa yok); p değeri verilmedi (betimsel)."
    if item_on_bound(f_alt, i) or item_on_bound(f_null, i):
        return stat, False, ("Maddenin bir parametresi kestirim sınırında (örn. c = 0); χ² referansı geçerli değil, "
                             "p değeri verilmedi (betimsel).")
    if not (f_alt["regular"] and f_null["regular"]):
        return stat, False, ("Başka maddelerde veya gizil dağılımda sınır parametresi var; χ² referansı "
                             "varsayılamaz, p değeri verilmedi (betimsel).")
    return stat, True, ""


def _contrast(f, i, p):
    d = f["design"]; r, s = d.idx[p][0, i], d.idx[p][1, i]
    if r < 0 or r == s:
        return np.nan, np.nan
    delta = float(f["params"][s] - f["params"][r])
    if f["cov"] is None or f["at_bound"][r] or f["at_bound"][s]:
        return delta, np.nan
    c = f["cov"]
    return delta, float(np.sqrt(max(c[r, r] + c[s, s] - 2 * c[r, s], 0.)))


def response_function_comparison(ref, foc, mu_f, sd_f, points=41, tol=1e-3):
    """Focal-weighted signed/unsigned P_F - P_R and crossings inside mu_F +- 1.96 sigma_F."""
    z, lw = _grid(points); w = np.exp(lw); theta = mu_f + sd_f * z
    pr = irf(theta, *[np.array([v]) for v in ref])[0][:, 0]; pf = irf(theta, *[np.array([v]) for v in foc])[0][:, 0]
    diff = pf - pr
    grid = np.linspace(mu_f - 1.96 * sd_f, mu_f + 1.96 * sd_f, 801)
    dg = irf(grid, *[np.array([v]) for v in foc])[0][:, 0] - irf(grid, *[np.array([v]) for v in ref])[0][:, 0]
    big = np.abs(dg) > tol
    sgn = np.sign(dg[big]); crossings = []
    if big.any():
        idx = np.where(np.diff(sgn) != 0)[0]
        g_big = grid[big]
        crossings = [float((g_big[j] + g_big[j + 1]) / 2) for j in idx]
    if not big.any():
        shape = "none"
    elif crossings:
        shape = "crossing"
    else:
        shape = "against" if sgn[0] < 0 else "favour"
    return {"signed": float(w @ diff), "unsigned": float(w @ np.abs(diff)), "crossings": crossings,
            "shape": shape, "max_abs": float(np.abs(dg).max())}


def dif_analysis_34(data, group, reference, focal, model="3PL", alpha=.05, purify=True, max_rounds=10,
                    min_anchors=None, quadrature="gauss-hermite", points=41, min_group=300, bounds=None,
                    inferential=True):
    if model not in PARAMS:
        raise ValueError("Bu DIF uygulaması yalnızca 3PL ve 4PL içindir.")
    if reference == focal:
        raise ValueError("Referans ve odak grup farklı olmalıdır.")
    if not 0 < alpha < .5:
        raise ValueError("Anlamlılık düzeyi 0 ile 0,5 arasında olmalıdır.")
    x, obs, groups, keep = _prepare(data, group, reference, focal)
    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"M{i+1}" for i in range(x.shape[1])]
    k = x.shape[1]; npar = len(PARAMS[model])
    sizes = {reference: int((groups == 0).sum()), focal: int((groups == 1).sum())}
    if min(sizes.values()) < 100:
        raise ValueError(f"{model} DIF için her grupta en az 100 kişi gereklidir (bulunan: {sizes}).")
    warnings = []
    if min(sizes.values()) < min_group:
        warnings.append(f"Grup büyüklükleri küçük ({sizes}); {model} parametreleri (özellikle c"
                        + (" ve d" if model == "4PL" else "") + ") zayıf tanımlanır, DIF kestirimleri kararsızdır.")
    ratio = max(sizes.values()) / min(sizes.values())
    if ratio > 4:
        warnings.append(f"Grup büyüklükleri dengesiz (oran {ratio:.1f}); küçük grubun parametreleri belirsizdir.")
    testable, untestable = [], {}
    for i in range(k):
        reasons = []
        for g, label in ((0, reference), (1, focal)):
            o = obs[groups == g, i]; v = x[groups == g, i][o]
            if o.sum() == 0:
                reasons.append(f"{label} grubunda hiç yanıt yok")
            elif v.min() == v.max():
                reasons.append(f"{label} grubunda tüm yanıtlar aynı")
        (untestable.__setitem__(i, "; ".join(reasons)) if reasons else testable.append(i))
    linking = [i for i in range(k) if obs[groups == 0, i].any() and obs[groups == 1, i].any()]
    if len(linking) < 2:
        raise ValueError("Bağlantısız tasarım: iki grubun ortak yanıtladığı madde sayısı yetersiz.")
    if untestable:
        warnings.append("DIF için test edilemeyen maddeler (yalnızca çapa olabilir): "
                        + "; ".join(f"{names[i]}: {r}" for i, r in untestable.items()))
    min_anchors = max(2, math.ceil(.2 * k)) if min_anchors is None else int(min_anchors)
    fitter = _Fitter34(x, obs, groups, model, quadrature, points, bounds or default_bounds())

    def types_for(pool, i, role):
        t = ["anchor" if (j in pool or j not in testable) else "all" for j in range(k)]
        t[i] = role
        return t

    def omnibus(pool, information=False):
        rows = {}
        for i in testable:
            pl = [j for j in pool if j != i and j in linking]
            if not pl:
                rows[i] = {"note": "Çapa yok", "ok": False, "stat": np.nan}; continue
            m0 = fitter(types_for(pl, i, "anchor")); m_all = fitter(types_for(pl, i, "all"), information)
            stat, ok, note = _lr(m_all, m0, i)
            rows[i] = {"stat": stat, "ok": ok and inferential, "note": note, "m0": m0, "m_all": m_all, "pool": pl}
        return rows

    critical = chi2.ppf(1 - alpha, npar)

    def flags(rows):
        """BH-flagged items (inferential) and screened items (no p-value, LR above the chi-square
        critical value): the latter only leave the anchor set (heuristic), they are not 'DIF'."""
        p = np.array([chi2.sf(rows[i]["stat"], npar) if rows[i]["ok"] else np.nan for i in testable])
        q = benjamini_hochberg(p)
        flagged = {testable[j] for j in range(len(testable)) if np.isfinite(q[j]) and q[j] < alpha}
        screened = {i for i in testable if not rows[i]["ok"] and np.isfinite(rows[i]["stat"])
                    and rows[i]["stat"] >= critical}
        return p, q, flagged, screened

    history = []; stable = True
    rows = omnibus(list(range(k)))
    p, q, flagged, screened = flags(rows)
    removed = flagged | screened if inferential else set()
    history.append({"Tur": 0, "Çapa sayısı": k - 1, "İşaretlenen": sorted(names[i] for i in flagged),
                    "Taramayla çapa dışı": sorted(names[i] for i in screened - flagged)})
    pool = list(range(k))
    if purify and inferential:
        stable = False
        for r in range(1, max_rounds + 1):
            if not removed:
                stable = True; break
            pool = [j for j in range(k) if j not in removed]
            if len(pool) < min_anchors:
                warnings.append(f"Arındırma sonrası değişmez kabul edilen madde sayısı ({len(pool)}) önerilen alt "
                                f"sınırın ({min_anchors}) altında; çapa kümesi kararsız.")
                break
            rows = omnibus(pool)
            p, q, flagged, screened = flags(rows)
            new = flagged | screened
            history.append({"Tur": r, "Çapa sayısı": len(pool), "İşaretlenen": sorted(names[i] for i in flagged),
                            "Taramayla çapa dışı": sorted(names[i] for i in screened - flagged)})
            if new == removed:
                stable = True; break
            removed = new
        if not stable:
            warnings.append("Çapa arındırma kararlı bir kümeye ulaşmadı; sonuçlar çapa seçimine duyarlıdır.")
    rows = omnibus(pool, information=True)
    p, q, flagged, screened = flags(rows)
    removed = (flagged | screened) if inferential else set()
    no_p = [names[i] for i in testable if not rows[i]["ok"]]
    if no_p and inferential:
        warnings.append("Genel DIF testi p değeri verilemeyen maddeler (çapa olarak kalırlar; DIF içeriyorlarsa "
                        "bağlantılamayı etkileyebilir): " + ", ".join(no_p))
    final_anchors = [j for j in range(k) if j not in removed]
    if screened - flagged and inferential:
        warnings.append("p değeri verilemeyen ancak LR istatistiği χ² kritik değerini aşan maddeler çapa kümesinden "
                        "çıkarıldı (sezgisel tarama; DIF kanıtı olarak yorumlanmaz): "
                        + ", ".join(names[i] for i in sorted(screened - flagged)))
    if len(final_anchors) < min_anchors:
        warnings.append(f"Değişmez madde sayısı ({len(final_anchors)}) önerilen alt sınırın ({min_anchors}) altında.")
    pi, qi = dict(zip(testable, p)), dict(zip(testable, q))
    joint = fitter(["anchor" if (j in final_anchors or j not in testable) else "all" for j in range(k)])
    pars = PARAMS[model]
    table = []
    for i in range(k):
        base = {"Madde": names[i], "Çapa": i in final_anchors}
        r = rows.get(i, {})
        if "m_all" not in r:
            base.update({"DIF işareti": False, "Çıkarım": "Test edilemedi",
                         "Not": untestable.get(i, r.get("note", ""))})
            table.append(base); continue
        m0, ma = r["m0"], r["m_all"]; sig = bool(np.isfinite(qi[i]) and qi[i] < alpha)
        for prm in pars:
            dlt, se = _contrast(ma, i, prm)
            base[f"Δ{prm} (odak − ref)"] = dlt; base[f"SH(Δ{prm})"] = se
        base.update({f"LR χ² genel (sd={npar})": r["stat"], "p": pi[i], "q (BH)": qi[i], "DIF işareti": sig,
                     "Çapa taraması": ("Çapa dışı (sezgisel)" if i in screened and not sig and inferential else ""),
                     "Çıkarım": ("Asimptotik χ² (düzenli çözüm)" if r["ok"] else
                                 "Betimsel (p verilmedi)" if inferential else "Betimsel")})
        notes = [r["note"]] if r["note"] else []
        if sig or (i in screened and inferential):        # parameter-specific decomposition
            for prm in pars:
                mp = fitter(types_for(r["pool"], i, prm))
                stat, ok, note = _lr(mp, m0, i)
                base[f"LR χ² Δ{prm} (sd=1)"] = stat
                base[f"p (Δ{prm})"] = chi2.sf(stat, 1) if ok else np.nan
        if ma["converged"]:
            d = ma["design"]
            ref = [d.unpack(ma["params"], 0)[j][i] for j in range(4)]
            foc = [d.unpack(ma["params"], 1)[j][i] for j in range(4)]
            s = response_function_comparison(ref, foc, ma["mu_F"], ma["sd_F"], points)
            base.update({"İşaretli fark E_F[P_F − P_R]": s["signed"], "İşaretsiz fark E_F|P_F − P_R|": s["unsigned"],
                         "En büyük |P_F − P_R|": s["max_abs"], "Tepki fonksiyonu karşılaştırması": SHAPES[s["shape"]],
                         "Kesişim θ*": ", ".join(f"{v:.2f}" for v in s["crossings"])})
            for j, prm in enumerate("abcd"):
                base[f"{prm}_ref"] = ref[j]; base[f"{prm}_odak"] = foc[j]
            if ma["cov"] is None and ma["se_reason"]:
                notes.append(ma["se_reason"])
            elif ma.get("cov_conditional"):
                notes.append("SH'ler koşullu bilgi matrisinden.")
        base["Not"] = " ".join(n for n in notes if n)
        table.append(base)
    return {"table": pd.DataFrame(table), "anchors": [names[j] for j in final_anchors],
            "history": pd.DataFrame(history), "stable": stable, "joint": joint, "sizes": sizes,
            "reference": reference, "focal": focal, "alpha": alpha, "purify": bool(purify and inferential),
            "min_anchors": min_anchors, "quadrature": quadrature, "points": points, "model": model,
            "inferential": bool(inferential), "warnings": warnings, "kept_rows": keep, "n_fits": len(fitter.cache),
            "estimand": ("Δp = p_odak − p_referans (p ∈ {a, b, c" + (", d" if model == "4PL" else "") + "}); "
                         "işaretli fark < 0: madde odak grup için daha zor. DIF'in biçimi grup tepki "
                         "fonksiyonlarından belirlenir.")}
