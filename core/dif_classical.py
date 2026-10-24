"""Observed-score DIF: Mantel–Haenszel, Mantel (ordinal), logistic and ordinal logistic regression.

Binary items
  * Mantel–Haenszel (Holland & Thayer, 1988): stratified by the matching score; chi-square with
    continuity correction (1 df), common odds ratio alpha_MH, Robins–Breslow–Greenland SE of
    ln(alpha_MH), Wald 95% CI, ETS delta = -2.35 ln(alpha_MH) and ETS A/B/C classes.
  * Logistic regression (Swaminathan & Rogers, 1990): M0 score, M1 + group, M2 + score x group.
    Uniform LR = M1 vs M0 (1 df), nonuniform = M2 vs M1 (1 df), omnibus = M2 vs M0 (2 df);
    Nagelkerke Delta R^2 with Jodoin & Gierl (2001) A/B/C (.035 / .070).
Ordinal items
  * Mantel (1963) 1-df chi-square for ordered scores and the standardized mean difference (SMD,
    Dorans & Schmitt, 1991; focal-weighted).
  * Ordinal (cumulative-logit) logistic regression with the same three nested models.
Matching is the total score of the matching items INCLUDING the studied item (Zwick, 2012);
optional iterative purification removes flagged items from the matching score (studied item kept).
p-values are adjusted within the item family by BH (default), BY, Holm or Bonferroni.
These are observed-score methods; under group impact with unequal discriminations they can show
inflated Type I error (documented in validation/evidence/p2_dif). Not IRT likelihood DIF.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

ADJUST_METHODS = ("BH", "BY", "Holm", "Bonferroni", "none")
ETS_B, ETS_C = 1.0, 1.5
JG_B, JG_C = .035, .070


def adjust_pvalues(p, method="BH"):
    """Family-wise / false-discovery adjustment; NaN entries are excluded and kept NaN."""
    if method not in ADJUST_METHODS:
        raise ValueError(f"Bilinmeyen düzeltme yöntemi: {method}")
    p = np.asarray(p, float); out = np.full(p.shape, np.nan)
    ok = np.isfinite(p); m = int(ok.sum())
    if not m:
        return out
    v = p[ok]
    if method == "none":
        adj = v.copy()
    elif method == "Bonferroni":
        adj = np.minimum(v * m, 1.)
    elif method == "Holm":
        order = np.argsort(v); ranked = v[order] * (m - np.arange(m))
        q = np.minimum(np.maximum.accumulate(ranked), 1.)
        adj = np.empty(m); adj[order] = q
    else:
        c = 1. if method == "BH" else float(np.sum(1. / np.arange(1, m + 1)))
        order = np.argsort(v); ranked = v[order] * m * c / np.arange(1, m + 1)
        q = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.)
        adj = np.empty(m); adj[order] = q
    out[ok] = adj
    return out


def _prepare(data, group, reference, focal, ordinal):
    frame = pd.DataFrame(data).apply(pd.to_numeric, errors="coerce")
    g = pd.Series(group).reset_index(drop=True)
    frame = frame.reset_index(drop=True)
    if len(g) != len(frame):
        raise ValueError("Grup değişkeni veriyle aynı uzunlukta olmalıdır.")
    if reference == focal:
        raise ValueError("Referans ve odak grup farklı olmalıdır.")
    keep = g.isin([reference, focal]).to_numpy()
    x = frame.loc[keep]
    if x.isna().any().any():
        raise ValueError("Gözlenen puan DIF yöntemleri eksiksiz yanıt gerektirir; eksik yanıt sessizce doldurulmaz.")
    x = x.to_numpy(float)
    if not np.array_equal(x, np.floor(x)) or (x < 0).any():
        raise ValueError("Yanıtlar negatif olmayan tam sayı puanlar olmalıdır.")
    if not ordinal and not np.isin(x, [0, 1]).all():
        raise ValueError("İkili DIF yöntemleri 0/1 yanıt gerektirir.")
    gf = (g[keep] == focal).to_numpy(int)
    if gf.sum() < 20 or (1 - gf).sum() < 20:
        raise ValueError("Her grupta en az 20 kişi gereklidir.")
    if x.shape[1] < 3:
        raise ValueError("En az 3 madde gereklidir.")
    return x, gf, [str(c) for c in frame.columns]


# ---------------------------------------------------------------- Mantel–Haenszel (binary)
def mantel_haenszel(y, gf, match):
    """Single-item MH statistics. gf: 1 = focal, 0 = reference."""
    A = B = C = D = 0.
    sumA = sumEA = sumVA = 0.
    R = S = PR = PSQR = QS = 0.
    used = 0
    for s in np.unique(match):
        ix = match == s
        r, f = ix & (gf == 0), ix & (gf == 1)
        nR, nF = r.sum(), f.sum(); T = nR + nF
        if nR == 0 or nF == 0 or T < 2:
            continue
        a = y[r].sum(); b = nR - a; c = y[f].sum(); d = nF - c
        m1 = a + c; m0 = T - m1
        if m1 == 0 or m0 == 0:
            continue
        used += 1
        sumA += a; sumEA += nR * m1 / T; sumVA += nR * nF * m1 * m0 / (T * T * (T - 1))
        Rk, Sk = a * d / T, b * c / T; P, Q = (a + d) / T, (b + c) / T
        R += Rk; S += Sk; PR += P * Rk; PSQR += P * Sk + Q * Rk; QS += Q * Sk
    out = {"strata_used": used}
    if used == 0 or sumVA <= 0:
        out.update(chi2=np.nan, p=np.nan, alpha_mh=np.nan, log_or=np.nan, se_log_or=np.nan,
                   delta=np.nan, reason="Her iki grubu ve iki yanıtı içeren eşleştirme katmanı yok.")
        return out
    stat = (max(abs(sumA - sumEA) - .5, 0.)) ** 2 / sumVA
    out.update(chi2=float(stat), p=float(chi2.sf(stat, 1)))
    if R <= 0 or S <= 0:
        out.update(alpha_mh=np.nan, log_or=np.nan, se_log_or=np.nan, delta=np.nan,
                   reason="Ortak olasılık oranı tanımsız (sıfır hücre toplamı).")
        return out
    lor = float(np.log(R / S))
    var = PR / (2 * R * R) + PSQR / (2 * R * S) + QS / (2 * S * S)
    se = float(np.sqrt(var))
    out.update(alpha_mh=float(R / S), log_or=lor, se_log_or=se, delta=-2.35 * lor, reason="")
    return out


def ets_mh_class(delta, se_log_or, p, alpha=.05):
    """ETS A/B/C (Zwick, 2012): C if |D| >= 1.5 and |D| significantly > 1 (one-sided .05)."""
    if not np.isfinite(delta) or not np.isfinite(p) or p >= alpha or abs(delta) < ETS_B:
        return "A"
    se_d = 2.35 * se_log_or
    if abs(delta) >= ETS_C and np.isfinite(se_d) and (abs(delta) - ETS_B) / se_d > norm.ppf(.95):
        return "C"
    return "B"


# ---------------------------------------------------------------- Mantel (ordinal) + SMD
def mantel_ordinal(y, gf, match):
    F = EF = VF = 0.; smd_num = 0.; nF_total = (gf == 1).sum(); used = 0
    for s in np.unique(match):
        ix = match == s
        r, f = ix & (gf == 0), ix & (gf == 1)
        nR, nF = r.sum(), f.sum(); T = nR + nF
        if nR == 0 or nF == 0 or T < 2:
            continue
        ys = y[ix]; sy, sy2 = ys.sum(), (ys ** 2).sum()
        v = nR * nF / (T * T * (T - 1)) * (T * sy2 - sy * sy)
        if v <= 0:
            continue
        used += 1
        F += y[f].sum(); EF += nF * sy / T; VF += v
        smd_num += nF * (y[f].mean() - y[r].mean())
    if used == 0 or VF <= 0:
        return {"chi2": np.nan, "p": np.nan, "smd": np.nan, "strata_used": used,
                "reason": "Kullanılabilir eşleştirme katmanı yok."}
    stat = (F - EF) ** 2 / VF
    return {"chi2": float(stat), "p": float(chi2.sf(stat, 1)), "smd": float(smd_num / nF_total),
            "strata_used": used, "reason": ""}


# ---------------------------------------------------------------- logistic regression
def _nagelkerke(ll, ll0, n):
    num = 1 - np.exp(2 * (ll0 - ll) / n); den = 1 - np.exp(2 * ll0 / n)
    return float(num / den) if den > 0 else np.nan


def _logit_ll(y, X):
    import statsmodels.api as sm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.Logit(y, X).fit(method="newton", maxiter=100, disp=False)
    if not res.mle_retvals.get("converged", False) or not np.isfinite(res.llf):
        raise RuntimeError("Lojistik model yakınsamadı (olası tam ayrışma).")
    return float(res.llf), res


def _ordinal_ll(y, X):
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=False, maxiter=500)
    if not np.isfinite(res.llf) or not res.mle_retvals.get("converged", False):
        raise RuntimeError("Sıralı lojistik model yakınsamadı.")
    return float(res.llf), res


def logistic_dif(y, gf, match, ordinal=False):
    """Three nested models; returns LR statistics, p-values, Nagelkerke R^2 and effects."""
    n = len(y)
    z = (match - match.mean()) / (match.std() if match.std() > 0 else 1.)
    cols = {"M0": np.column_stack([z]), "M1": np.column_stack([z, gf]), "M2": np.column_stack([z, gf, z * gf])}
    try:
        if ordinal:
            yy = pd.Series(pd.Categorical(y.astype(int), ordered=True))
            ll = {m: _ordinal_ll(yy, X)[0] for m, X in cols.items()}
            counts = np.bincount(y.astype(int)); counts = counts[counts > 0]
            ll_null = float(np.sum(counts * np.log(counts / n)))
            beta_group = np.nan
        else:
            import statsmodels.api as sm
            fits = {m: _logit_ll(y, sm.add_constant(X, has_constant="add")) for m, X in cols.items()}
            ll = {m: v[0] for m, v in fits.items()}
            p1 = y.mean(); ll_null = float(n * (p1 * np.log(p1) + (1 - p1) * np.log(1 - p1)))
            beta_group = float(fits["M1"][1].params[2])
    except Exception as exc:
        return {"reason": str(exc), **{k: np.nan for k in ("lr_uniform", "p_uniform", "lr_nonuniform", "p_nonuniform",
                                                              "lr_omnibus", "p_omnibus", "delta_r2", "beta_group")}}
    r2 = {m: _nagelkerke(v, ll_null, n) for m, v in ll.items()}
    lr_u, lr_n, lr_o = 2 * (ll["M1"] - ll["M0"]), 2 * (ll["M2"] - ll["M1"]), 2 * (ll["M2"] - ll["M0"])
    lr_u, lr_n, lr_o = (max(v, 0.) for v in (lr_u, lr_n, lr_o))
    return {"lr_uniform": lr_u, "p_uniform": float(chi2.sf(lr_u, 1)), "lr_nonuniform": lr_n,
            "p_nonuniform": float(chi2.sf(lr_n, 1)), "lr_omnibus": lr_o, "p_omnibus": float(chi2.sf(lr_o, 2)),
            "delta_r2": r2["M2"] - r2["M0"], "delta_r2_uniform": r2["M1"] - r2["M0"], "beta_group": beta_group,
            "reason": ""}


def jodoin_gierl(delta_r2, p, alpha=.05):
    if not np.isfinite(delta_r2) or not np.isfinite(p) or p >= alpha or delta_r2 < JG_B:
        return "A"
    return "C" if delta_r2 >= JG_C else "B"


# ---------------------------------------------------------------- item-family analysis
def dif_screen(data, group, reference, focal, *, method="MH", ordinal=False, adjust="BH", alpha=.05,
               purify=True, max_rounds=10):
    """Run MH/Mantel or LR for all items with optional iterative purification of the matching score."""
    if method not in ("MH", "LR"):
        raise ValueError("Yöntem MH veya LR olmalıdır.")
    x, gf, names = _prepare(data, group, reference, focal, ordinal)
    k = x.shape[1]
    matching = np.ones(k, bool); history = []

    def run(match_items):
        rows = []
        base = x[:, match_items].sum(1)
        for j in range(k):
            m = base + (0 if match_items[j] else x[:, j])  # studied item always included
            y = x[:, j]
            if method == "MH":
                r = mantel_ordinal(y, gf, m) if ordinal else mantel_haenszel(y, gf, m)
                p = r["p"]
            else:
                r = logistic_dif(y, gf, m, ordinal=ordinal); p = r["p_omnibus"]
            rows.append({"Madde": names[j], "p": p, **r})
        frame = pd.DataFrame(rows)
        frame["p_adj"] = adjust_pvalues(frame["p"], adjust)
        return frame

    frame = run(matching); rounds = 1
    if purify:
        for rounds in range(1, max_rounds + 1):
            flagged = (frame["p_adj"] < alpha).to_numpy()
            new = ~flagged
            if new.sum() < 2:
                history.append("Saflaştırma durduruldu: eşleştirme için 2'den az madde kalırdı.")
                break
            if np.array_equal(new, matching):
                break
            matching = new; frame = run(matching)
        else:
            history.append("Saflaştırma azami tur sayısında durağanlaşmadı; sonuçlar son tura aittir.")
    frame["Eşleştirmede"] = matching
    frame["DIF (düzeltilmiş p < alpha)"] = frame["p_adj"] < alpha
    if method == "MH" and not ordinal:
        frame["ETS sınıfı"] = [ets_mh_class(d, s, p, alpha) for d, s, p in
                               zip(frame["delta"], frame["se_log_or"], frame["p"])]
        z = norm.ppf(.975)
        frame["lnOR %95 alt"] = frame["log_or"] - z * frame["se_log_or"]
        frame["lnOR %95 üst"] = frame["log_or"] + z * frame["se_log_or"]
    if method == "LR":
        frame["Jodoin–Gierl sınıfı"] = [jodoin_gierl(d, p, alpha) for d, p in zip(frame["delta_r2"], frame["p_adj"])]
    tot = x.sum(1); r_, f_ = tot[gf == 0], tot[gf == 1]
    pooled = np.sqrt(((len(r_) - 1) * r_.var(ddof=1) + (len(f_) - 1) * f_.var(ddof=1)) / (len(tot) - 2))
    impact_d = float((f_.mean() - r_.mean()) / pooled) if pooled > 0 else np.nan
    if np.isfinite(impact_d) and abs(impact_d) >= .3:
        history.append(
            f"Gruplar arasında toplam puan farkı büyük (Cohen d = {impact_d:.2f}). Önceden tanımlı simülasyonda "
            "(validation/evidence/p2_dif) odak ortalaması −0,5 iken " +
            ("sıralı maddelerde aile bazlı hata (Holm FWER / BH FDR) %9–12'ye çıktı; sonuçları MTK-LR DIF ile "
             "doğrulayın." if ordinal else
             "ikili maddelerde madde başına I. tip hata %4–7, aile bazlı hata %4–9 bulundu; sınırda sonuçları "
             "MTK-LR DIF ile doğrulayın."))
    meta = {"method": method, "ordinal": ordinal, "adjust": adjust, "alpha": alpha, "purify": purify,
            "impact_cohen_d": impact_d,
            "rounds": rounds, "notes": history, "n_reference": int((gf == 0).sum()), "n_focal": int((gf == 1).sum())}
    return frame, meta
