"""Two-group DIF for the 1PL and 2PL models on a common metric (B10).

Model: concurrent two-group MML (core.irt_mg). Reference group theta ~ N(0, 1) (the
1PL/2PL identification of core.irt); focal mean mu_F and SD sigma_F free; anchor items
have equal parameters in both groups and link the groups.

Tests for a studied item i (all other items keep their current role):
    M0  item i constrained equal (anchor-like)
    M1  b_i group-specific; a_i common (2PL) or fixed at 1 (1PL)
    M2  a_i and b_i group-specific (2PL only)
1PL : LR(M1 vs M0), 1 df  -> uniform DIF in b (with a = 1 non-uniform DIF is not
      defined: group response functions cannot cross because no item-specific slope is
      estimated; use the 2PL page for slope differences, which is a different model).
2PL : omnibus LR(M2 vs M0), 2 df (used for flagging and BH-FDR); uniform LR(M1 vs M0),
      1 df; non-uniform LR(M2 vs M1), 1 df. LR statistics are invariant to the (a, b) vs
      slope-intercept parameterization. Wald contrasts with SEs from the joint observed
      information are reported for interpretation: 1PL Delta_b = b_F - b_R (from M1);
      2PL Delta_a and Delta_b from M2.

Uniform / non-uniform is decided from the group response functions, not from parameter
names: with a_F = a_R the logit difference a (b_R - b_F) is constant in theta (no
crossing); with a_F != a_R the curves cross at theta* = (a_F b_F - a_R b_R)/(a_F - a_R).
Effect sizes on the probability scale over the focal ability distribution:
signed  E_F[P_F(theta) - P_R(theta)]  (negative: harder for the focal group)
unsigned E_F|P_F(theta) - P_R(theta)|.

Anchors: all-other-items start, then iterative purification (omnibus test, BH-FDR).
p-values are conditional on the data-selected anchor set (selection not in the SE or
the LR reference distribution). p-values are withheld when a fit did not converge or a
parameter is on a bound; Wald SEs are withheld when the information is not positive
definite. Significant DIF is not proof of item bias.
"""
import math
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from scipy.special import expit
from core.irt_mg import as_matrix, Design, fit as mg_fit
from core.rasch import _grid
from core.rasch_dif import benjamini_hochberg, ets_class

DIF_TYPES = {"uniform": "Tek biçimli (uniform)",
             "crossing": "Tek biçimli olmayan — kesişimli",
             "noncrossing": "Tek biçimli olmayan — kesişimsiz (yön sabit, büyüklük θ'ya göre değişir)"}


def _prepare(data, group, reference, focal):
    x, obs = as_matrix(data)
    g = pd.Series(group).reset_index(drop=True)
    if len(g) != len(x):
        raise ValueError("Grup değişkeni ile veri satır sayısı uyuşmuyor.")
    keep = g.isin([reference, focal]).to_numpy() & obs.any(axis=1)
    return x[keep], obs[keep], (g[keep] == focal).to_numpy().astype(int), keep


class _Fitter:
    """Caches fits by the tuple of item constraint types."""

    def __init__(self, x, obs, groups, model, quadrature, points):
        self.args = (x, obs, groups); self.model = model; self.quadrature = quadrature; self.points = points
        self.cache = {}; self.base = None

    def __call__(self, types, information=False):
        key = (tuple(types), information)
        if key not in self.cache:
            if not information and (tuple(types), True) in self.cache:
                return self.cache[(tuple(types), True)]
            design = Design(self.model, types, 2)
            done = self.cache.get((tuple(types), False))
            start = done["params"] if done is not None else self._start(design)
            self.cache[key] = mg_fit(*self.args, design, self.quadrature, self.points, start=start,
                                     information=information)
            if self.base is None:
                self.base = self.cache[key]
        return self.cache[key]

    def _start(self, design):
        if self.base is None:
            return None
        bd = self.base["design"]; start = np.zeros(design.size)
        for j, (kind, g, i) in enumerate(design.labels):
            gg = max(g, 0)
            src = bd.a[gg, i] if kind == "a" else bd.b[gg, i]
            start[j] = self.base["params"][src] if src >= 0 else 1.
        start[design.latent:] = self.base["params"][bd.latent:]
        return start


def _valid(*fits):
    return all(f["converged"] for f in fits)


def _lr(f_alt, f_null):
    if not _valid(f_alt, f_null):
        return np.nan
    return _nested_lr(f_alt["loglik"], f_null["loglik"])[0]


def _nested_lr(loglik_alt, loglik_null):
    """Reject failed nested optimization rather than turn it into evidence of no DIF.

    At a nested maximum log L_alt >= log L_null. Allow only 1e-6 absolute
    log-likelihood units of numerical roundoff; do not scale this tolerance by N.
    This check is necessary, but does not establish convergence or regularity.
    """
    if not (np.isfinite(loglik_alt) and np.isfinite(loglik_null)):
        return np.nan, "Sonlu olmayan log-olabilirlik; p değeri verilmedi."
    difference = loglik_alt - loglik_null
    if difference < -1e-6:
        return np.nan, ("Alternatif modelin log-olabilirliği kısıtlı modelden düşük; "
                        "optimizasyon doğrulanamadı, p değeri verilmedi.")
    return max(2. * difference, 0.), ""


def _contrast(f, i, kind):
    d = f["design"]; idx = d.a if kind == "a" else d.b
    r, s = idx[0, i], idx[1, i]
    if r < 0 or r == s:
        return 0., np.nan
    delta = float(f["params"][s] - f["params"][r])
    if f["cov"] is None:
        return delta, np.nan
    c = f["cov"]
    return delta, float(np.sqrt(max(c[r, r] + c[s, s] - 2 * c[r, s], 0.)))


def response_function_summary(a_r, b_r, a_f, b_f, mu_f, sd_f, points=41):
    """Signed/unsigned focal-weighted P_F - P_R and crossing point of the two IRFs."""
    z, lw = _grid(points); theta = mu_f + sd_f * z; w = np.exp(lw)
    diff = expit(a_f * (theta - b_f)) - expit(a_r * (theta - b_r))
    if abs(a_f - a_r) < 1e-12:
        cross = np.nan
    else:
        cross = (a_f * b_f - a_r * b_r) / (a_f - a_r)
    lo, hi = mu_f - 1.96 * sd_f, mu_f + 1.96 * sd_f
    return {"signed": float(w @ diff), "unsigned": float(w @ np.abs(diff)), "crossing": float(cross),
            "crossing_inside": bool(np.isfinite(cross) and lo <= cross <= hi), "focal_range": (lo, hi)}


def dif_analysis(data, group, reference, focal, model="2PL", alpha=.05, purify=True, max_rounds=10,
                 min_anchors=None, quadrature="gauss-hermite", points=41, min_group=200):
    if model not in ("1PL", "2PL"):
        raise ValueError("Bu DIF uygulaması yalnızca 1PL ve 2PL içindir.")
    if reference == focal:
        raise ValueError("Referans ve odak grup farklı olmalıdır.")
    if not 0 < alpha < .5:
        raise ValueError("Anlamlılık düzeyi 0 ile 0,5 arasında olmalıdır.")
    x, obs, groups, keep = _prepare(data, group, reference, focal)
    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"M{i+1}" for i in range(x.shape[1])]
    k = x.shape[1]
    sizes = {reference: int((groups == 0).sum()), focal: int((groups == 1).sum())}
    if min(sizes.values()) < 50:
        raise ValueError(f"Her grupta en az 50 kişi gereklidir (bulunan: {sizes}).")
    warnings = []
    if min(sizes.values()) < min_group:
        warnings.append(f"Grup büyüklükleri küçük ({sizes}); {model} DIF kestirimleri kararsız ve testlerin gücü "
                        "düşük olabilir; LR testinin asimptotik χ² yaklaşımı zayıflayabilir.")
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
    free_type = "ab" if model == "2PL" else "b"
    fitter = _Fitter(x, obs, groups, model, quadrature, points)

    def types_for(anchor_pool, i, role):
        t = ["anchor" if (j in anchor_pool or j not in testable) else free_type for j in range(k)]
        t[i] = role
        return t

    def test_items(anchor_pool, information=False):
        rows = {}
        for i in testable:
            pool = [j for j in anchor_pool if j != i and j in linking]
            if not pool:
                rows[i] = {"note": "Çapa yok"}; continue
            m0 = fitter(types_for(pool, i, "anchor"))
            m1 = fitter(types_for(pool, i, "b"), information=information)
            m2 = fitter(types_for(pool, i, "ab"), information=information) if model == "2PL" else None
            if model == "1PL":
                lr = _lr(m1, m0); rows[i] = {"omni": lr, "df": 1, "uni": lr, "non": np.nan}
            else:
                rows[i] = {"omni": _lr(m2, m0), "df": 2, "uni": _lr(m1, m0), "non": _lr(m2, m1)}
            bad = [f for f in (m0, m1, m2) if f is not None and not f["converged"]]
            rows[i]["note"] = (bad[0]["se_reason"] + " p değeri verilmedi.") if bad else ""
            if not bad:
                comparisons = [("Tek biçimli", m1, m0)] if m2 is None else [
                    ("Genel", m2, m0), ("Tek biçimli", m1, m0), ("Tek biçimli olmayan", m2, m1)]
                notes = [f"{label}: {note}" for label, a, n in comparisons
                         if (note := _nested_lr(a["loglik"], n["loglik"])[1])]
                rows[i]["note"] = " ".join(dict.fromkeys(note for note in notes if note))
            rows[i]["fits"] = (m0, m1, m2)
        return rows

    def flags(rows):
        p = np.array([chi2.sf(rows[i]["omni"], rows[i]["df"]) if np.isfinite(rows[i].get("omni", np.nan)) else np.nan
                      for i in testable])
        q = benjamini_hochberg(p)
        return p, q, {testable[j] for j in range(len(testable)) if np.isfinite(q[j]) and q[j] < alpha}

    history = []
    rows = test_items(list(range(k)))
    p, q, flagged = flags(rows)
    history.append({"Tur": 0, "Çapa sayısı": k - 1, "İşaretlenen": sorted(names[i] for i in flagged)})
    stable = True
    anchors = [j for j in range(k) if j not in flagged]
    if purify:
        stable = False
        for r in range(1, max_rounds + 1):
            anchors = [j for j in range(k) if j not in flagged]
            if len(anchors) < min_anchors:
                warnings.append(f"Arındırma sonrası değişmez kabul edilen madde sayısı ({len(anchors)}) önerilen "
                                f"alt sınırın ({min_anchors}) altında; çapa kümesi kararsız.")
                break
            rows = test_items(anchors)
            p, q, new = flags(rows)
            history.append({"Tur": r, "Çapa sayısı": len(anchors), "İşaretlenen": sorted(names[i] for i in new)})
            if new == flagged:
                stable = True; break
            flagged = new
        if not stable:
            warnings.append("Çapa arındırma azami tur sayısında kararlı bir kümeye ulaşmadı; sonuçlar çapa "
                            "seçimine duyarlıdır.")
    final_anchors = [j for j in range(k) if j not in flagged]
    if len(final_anchors) < min_anchors:
        warnings.append(f"Değişmez madde sayısı ({len(final_anchors)}) önerilen alt sınırın ({min_anchors}) altında.")
    # Final round with information matrices for the Wald contrasts (anchor set of that round).
    pool_final = anchors if purify else list(range(k))
    rows = test_items(pool_final, information=True)
    p, q, flagged_final = flags(rows)
    pi, qi = dict(zip(testable, p)), dict(zip(testable, q))
    joint = fitter(["anchor" if (j in final_anchors or j not in testable) else free_type for j in range(k)],
                   information=True)
    table = []
    for i in range(k):
        base = {"Madde": names[i], "Çapa": i in final_anchors}
        if i not in rows or "fits" not in rows[i]:
            base.update({"Not": untestable.get(i, rows.get(i, {}).get("note", "")), "DIF işareti": False})
            table.append(base); continue
        r = rows[i]; m0, m1, m2 = r["fits"]; sig = bool(np.isfinite(qi[i]) and qi[i] < alpha)
        if model == "1PL":
            d_b, se_b = _contrast(m1, i, "b")
            base.update({"Δb (b_odak − b_ref)": d_b, "SH(Δb)": se_b,
                         "Wald z": d_b / se_b if np.isfinite(se_b) else np.nan,
                         "LR χ² (sd=1)": r["omni"], "p": pi[i], "q (BH)": qi[i], "DIF işareti": sig,
                         "Etki sınıfı (ETS benzeri)": ets_class(d_b, se_b, sig)})
            fa = m1
            ar = af = 1.
        else:
            d_a, se_a = _contrast(m2, i, "a"); d_b, se_b = _contrast(m2, i, "b")
            p_uni = chi2.sf(r["uni"], 1) if np.isfinite(r["uni"]) else np.nan
            p_non = chi2.sf(r["non"], 1) if np.isfinite(r["non"]) else np.nan
            base.update({"Δa (a_odak − a_ref)": d_a, "SH(Δa)": se_a, "Δb (b_odak − b_ref)": d_b, "SH(Δb)": se_b,
                         "LR χ² genel (sd=2)": r["omni"], "p": pi[i], "q (BH)": qi[i], "DIF işareti": sig,
                         "LR χ² tek biçimli (sd=1)": r["uni"], "p (tek biçimli)": p_uni,
                         "LR χ² tek biçimli olmayan (sd=1)": r["non"], "p (tek biçimli olmayan)": p_non})
            fa = m2
        if fa["converged"]:
            da = fa["design"]; prm = fa["params"]
            a_r, b_r = (da.unpack(prm, 0)[0][i], da.unpack(prm, 0)[1][i])
            a_f, b_f = (da.unpack(prm, 1)[0][i], da.unpack(prm, 1)[1][i])
            s = response_function_summary(a_r, b_r, a_f, b_f, fa["mu_F"], fa["sd_F"], points)
            base.update({"İşaretli fark E_F[P_F − P_R]": s["signed"], "İşaretsiz fark E_F|P_F − P_R|": s["unsigned"],
                         "a_ref": a_r, "b_ref": b_r, "a_odak": a_f, "b_odak": b_f})
            if model == "2PL":
                base["Kesişim θ*"] = s["crossing"] if sig else np.nan
                if sig:
                    kind = ("uniform" if not (np.isfinite(p_non) and p_non < alpha) else
                            "crossing" if s["crossing_inside"] else "noncrossing")
                    base["DIF türü"] = DIF_TYPES[kind]
                else:
                    base["DIF türü"] = ""
            elif sig:
                base["DIF türü"] = DIF_TYPES["uniform"]
        base["Not"] = r["note"] + ("" if fa["cov"] is not None or not fa["converged"] else " " + fa["se_reason"])
        table.append(base)
    table = pd.DataFrame(table)
    return {"table": table, "anchors": [names[j] for j in final_anchors], "history": pd.DataFrame(history),
            "stable": stable, "joint": joint, "sizes": sizes, "reference": reference, "focal": focal,
            "alpha": alpha, "purify": purify, "min_anchors": min_anchors, "quadrature": quadrature,
            "points": points, "model": model, "warnings": warnings, "kept_rows": keep, "n_fits": len(fitter.cache),
            "estimand": ("Δb = b_odak − b_referans (logit; a = 1). Δb > 0: madde aynı θ düzeyinde odak grup için "
                         "daha zor." if model == "1PL" else
                         "Δa = a_odak − a_referans, Δb = b_odak − b_referans; DIF türü grup tepki fonksiyonlarının "
                         "kesişimine göre belirlenir. İşaretli fark < 0: madde odak grup için daha zor.")}
