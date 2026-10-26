"""Yerel bağımsızlık ve madde uyumu raporu (iki kategorili MTK modelleri).

- Yerel bağımsızlık: Yen-Q3, merkezlenmiş Q3 (Q3 - ortalama Q3; Christensen vd.,
  2017) ve tek faktörlü PAF artık korelasyonları. Evrensel kesme puanı yoktur;
  isteğe bağlı parametrik bootstrap ile veri temelli kritik değer üretilir.
- Madde uyumu: Orlando-Thissen S-X² (sabit-parametre 2PL referans testi mevcut). Çıkarım politikası:
    * 1PL/2PL: asimptotik p ve BH-FDR keşifseldir; sabit-parametre
      referans kontrolü tam kestirim zincirinin sertifikası değildir.
    * 3PL/4PL veya yakınsamayan kestirim: p/FDR ve karar kapalıdır.
    * Çok kategorili modeller: S-X² uygulanmamıştır; analiz kapatılır.
  S-X², Pearson X² ve M2 farklı istatistiklerdir; burada yalnızca S-X² vardır.
"""
import numpy as np
import pandas as pd

from core.irt_python_mirt_adapter import fit_python_mirt
from core.irt_python_mirt_parameters import python_mirt_binary_parameters
from core.irt_itemfit import sx2_mirt_compatible
from core.irt_q3 import q3_analysis
from core.dimensionality import correlation_matrix, paf

ASYMPTOTIC_SX2 = ("1PL", "2PL")
SUPPORTED = ("1PL", "2PL", "3PL", "4PL")
EXPERIMENTAL = "EXPERIMENTAL / NOT CERTIFIED"


def bh_fdr(p):
    """Benjamini-Hochberg düzeltilmiş p (NaN korunur)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    m = np.isfinite(p)
    pv = p[m]
    if pv.size == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * pv.size / np.arange(1, pv.size + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty_like(pv); res[order] = np.minimum(adj, 1.0)
    out[m] = res
    return out


def item_fit(data, item_table, model, alpha=0.05, mincell=1.0, *, converged=True):
    if model not in SUPPORTED:
        raise ValueError("S-X² yalnızca iki kategorili 1PL/2PL/3PL/4PL için uygulanmıştır.")
    table, groups = sx2_mirt_compatible(data, item_table, model, mincell=mincell, return_tables=True)
    table["p (BH-FDR)"] = bh_fdr(table["p"])
    eligible = model in ASYMPTOTIC_SX2 and converged
    if eligible:
        table["Karar"] = np.where(table["p (BH-FDR)"].isna(), "Hesaplanamadı",
                                  np.where(table["p (BH-FDR)"] < alpha, "Uyumsuz (FDR)", "Uyumlu"))
        table["Sertifika"] = "NUMERICAL REFERENCE CHECKED / NOT CERTIFIED"
    else:
        table[["p", "p (BH-FDR)"]] = np.nan
        table["Karar"] = "Karar üretilmedi"
        table["Sertifika"] = EXPERIMENTAL
    return table, groups, False


def local_independence(data, item_table, model, person_method="WLE", bootstrap=0, seed=2026,
                       converged=True):
    a = item_table["a"].to_numpy(float); b = item_table["b"].to_numpy(float)
    kw = {}
    if model in ("3PL", "4PL"):
        kw["c"] = item_table["c"].to_numpy(float)
        kw["d"] = item_table["d"].to_numpy(float)
    def primary_refit(frame):
        result = fit_python_mirt(frame, model=model)
        params = python_mirt_binary_parameters(result.fit, model=model, item_names=list(frame.columns))
        values = tuple(params[name].to_numpy(float) for name in ("a", "b", "c", "d"))
        return (*values, bool(result.fit.converged)) if model in ("3PL", "4PL") else (*values[:2], bool(result.fit.converged))
    q3 = q3_analysis(data, a, b, model, person_method=person_method, bootstrap=bootstrap if converged else 0, refit=primary_refit,
                     seed=seed, converged=converged, **kw)
    cor = correlation_matrix(data, "auto")
    R = cor["R"].to_numpy()
    L, *_ = paf(R, 1)
    resid = R - np.outer(L[:, 0], L[:, 0]); np.fill_diagonal(resid, 0.0)
    names = list(data.columns)
    rmap = {(names[i], names[j]): resid[i, j] for i in range(len(names)) for j in range(i + 1, len(names))}
    pairs = q3["pairs"].copy()
    pairs["Artık korelasyon (1F)"] = [rmap.get((r["Madde 1"], r["Madde 2"]),
                                       rmap.get((r["Madde 2"], r["Madde 1"]))) for _, r in pairs.iterrows()]
    boot = q3.get("bootstrap")
    if isinstance(boot, dict) and boot.get("max_quantile_95") is not None:
        pairs["Bootstrap kritik değerini aşar"] = pairs["Düzeltilmiş Q3"] > boot["max_quantile_95"]
    return {"q3": q3, "pairs": pairs, "residuals": pd.DataFrame(resid, index=names, columns=names),
            "correlation_method": cor["method"]}


def run(data, model="2PL", alpha=0.05, person_method="WLE", bootstrap=0, seed=2026):
    """Tam yerel bağımsızlık + madde uyumu iş akışı (iki kategorili veri)."""
    data = data.apply(pd.to_numeric, errors="coerce")
    vals = np.unique(data.to_numpy(float)[np.isfinite(data.to_numpy(float))])
    if not set(vals).issubset({0.0, 1.0}):
        raise ValueError("Bu ekran yalnızca 0/1 puanlanmış veriyi destekler; çok kategorili "
                         "modeller için S-X² uygulanmamıştır.")
    res = fit_python_mirt(data, model=model)           # birincil kalibrasyon motoru (Python mirt)
    items = python_mirt_binary_parameters(res.fit, model=model, item_names=list(data.columns))
    converged = bool(getattr(res.fit, "converged", False))
    fit = {"converged": converged, "engine": res.engine, "engine_status": res.scientific_status}
    li = local_independence(data, items, model, person_method, bootstrap, seed, converged)
    fit_tab, groups, certified = item_fit(data.dropna(), items, model, alpha, converged=converged) if data.notna().all().all() \
        else (None, {}, False)
    note = None if fit_tab is not None else "S-X² eksik veri içeren matriste hesaplanmaz (tam veri gerekir)."
    return {"model": model, "fit": fit, "items": items, "local": li, "itemfit": fit_tab,
            "obs_exp": groups, "certified": certified, "note": note, "alpha": alpha}
