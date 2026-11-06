"""Inter-rater reliability and agreement: 30 preregistered methods (core/rater_methods.json).

Data model: WIDE = one row per rated unit, one column per rater (default); LONG = unit, rater, score;
COUNTS = one row per unit, one column per category (number of raters choosing it). The user's data are
never modified silently: problems are reported and analyses that cannot handle them fail closed.
"""
import itertools
import json
import math
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REGISTRY = json.loads((Path(__file__).with_name("rater_methods.json")).read_text(encoding="utf-8"))
METHODS = {m["id"]: m for m in REGISTRY["methods"]}
GROUPS = {"kategorik": "Kategorik Uyum", "sirali": "Sıralı ve Sürekli Puanlar", "surekli": "Sıralı ve Sürekli Puanlar",
          "sistematik": "Sistematik Puanlayıcı Farklılıkları"}
TWO_RATER = {"cohen_kappa", "cohen_kappa_linear", "cohen_kappa_quadratic", "scott_pi", "pabak", "bangdiwala_b",
             "spearman", "pearson", "mcnemar", "stuart_maxwell", "bhapkar", "bland_altman"}
MISSING_OK = {"kripp_nominal", "kripp_ordinal", "kripp_interval", "kripp_ratio"}
COUNTS_OK = {"fleiss_kappa", "gwet_ac1", "brennan_prediger", "percent_agreement"}
NUMERIC = {"cohen_kappa_linear", "cohen_kappa_quadratic", "gwet_ac2", "kripp_ordinal", "kripp_interval", "kripp_ratio",
           "icc_1_1", "icc_1_k", "icc_2_1", "icc_2_k", "icc_3_1", "icc_3_k", "kendall_w", "spearman", "pearson",
           "bland_altman"}
BINARY = {"pabak", "mcnemar"}
MIN_CATEGORIES = {"stuart_maxwell": 3, "bhapkar": 3}


class RaterDataError(ValueError):
    """Fail-closed input error with a Turkish explanation."""


@dataclass
class RaterData:
    layout: str                      # 'wide' or 'counts'
    table: pd.DataFrame              # wide: units x raters; counts: units x categories
    issues: list = field(default_factory=list)

    @property
    def n_units(self):
        return len(self.table)

    @property
    def n_raters(self):
        return int(self.table.sum(axis=1).iloc[0]) if self.layout == "counts" else self.table.shape[1]

    @property
    def has_missing(self):
        return self.layout == "wide" and bool(self.table.isna().to_numpy().any())

    def categories(self):
        if self.layout == "counts":
            return list(self.table.columns)
        values = pd.unique(self.table.to_numpy().ravel())
        values = [v for v in values if not _isna(v)]
        return sorted(values) if all(_is_number(v) for v in values) else sorted(values, key=str)

    @property
    def numeric(self):
        return self.layout == "wide" and all(_is_number(v) for v in self.categories())


def _isna(v):
    return v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA


def _is_number(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def _clean_cell(v, where, issues):
    """Return the cell as a number or a string exactly as given; record coding problems (never repaired)."""
    if _isna(v) or (isinstance(v, str) and v == ""):
        return np.nan
    if isinstance(v, (bool, np.bool_)):
        issues.append(("error", f"{where}: mantıksal değer ({v}) puan olarak kullanılamaz."))
        return np.nan
    if _is_number(v):
        f = float(v)
        if not math.isfinite(f):
            issues.append(("error", f"{where}: sonlu olmayan puan ({v})."))
            return np.nan
        return f
    s = str(v)
    if s != s.strip():
        issues.append(("error", f"{where}: '{s}' değerinde baştaki/sondaki boşluk farklı bir kod oluşturur; düzeltin."))
    try:
        float(s)
        issues.append(("error", f"{where}: sayı metin olarak kodlanmış ('{s}'); sütunu sayısal biçime çevirin."))
    except ValueError:
        pass
    return s


def _cells(frame, issues):
    out = frame.copy().astype(object)
    for c in out.columns:
        out[c] = [_clean_cell(v, f"'{c}' sütunu, satır {i + 1}", issues) for i, v in enumerate(frame[c].tolist())]
    kinds = {type(v) for v in out.to_numpy().ravel() if not _isna(v)}
    if float in kinds and str in kinds:
        issues.append(("error", "Sayısal ve metin kodlar birlikte kullanılmış; tek bir kodlama sistemi seçin."))
    lowered = {}
    for v in pd.unique(out.to_numpy().ravel()):
        if isinstance(v, str):
            lowered.setdefault(v.strip().casefold(), set()).add(v)
    for variants in lowered.values():
        if len(variants) > 1:
            issues.append(("error", "Aynı kategori farklı yazılmış: " + ", ".join(sorted(map(repr, variants))) + "."))
    return out


def from_wide(df, id_col, rater_cols):
    issues = []
    if id_col not in df or not rater_cols or any(c not in df for c in rater_cols):
        raise RaterDataError("Birim kimliği ve puanlayıcı sütunları veride bulunmalıdır.")
    if id_col in rater_cols:
        raise RaterDataError("Birim kimliği sütunu puanlayıcı olarak seçilemez.")
    if len(set(rater_cols)) != len(rater_cols):
        raise RaterDataError("Aynı puanlayıcı sütunu birden fazla seçilmiş.")
    ids = df[id_col]
    if ids.isna().any():
        issues.append(("error", "Birim kimliği boş olan satırlar var."))
    dup = ids[ids.duplicated()].astype(str).unique().tolist()
    if dup:
        issues.append(("error", "Tekrarlanan birim kimlikleri: " + ", ".join(dup[:10]) + ". Her satır tek bir puanlanan birim olmalıdır."))
    table = _cells(df[list(rater_cols)], issues)
    table.index = ids.astype(str).tolist()
    return _finish(RaterData("wide", table, issues))


def from_long(df, unit_col, rater_col, score_col):
    if any(c not in df for c in (unit_col, rater_col, score_col)) or len({unit_col, rater_col, score_col}) < 3:
        raise RaterDataError("Uzun biçimde birim, puanlayıcı ve puan için üç farklı sütun seçin.")
    issues = []
    if df[unit_col].isna().any() or df[rater_col].isna().any():
        raise RaterDataError("Birim veya puanlayıcı kimliği boş olamaz.")
    key = df[[unit_col, rater_col]].astype(str)
    dup = key[key.duplicated(keep=False)].drop_duplicates()
    if len(dup):
        pairs = ", ".join(f"{u}–{r}" for u, r in dup.itertuples(index=False))
        raise RaterDataError(f"Aynı birim–puanlayıcı için birden fazla puan var ({pairs}). Puanlar ortalanmaz; tekrarları kaldırın.")
    cells = _cells(df[[score_col]], issues)[score_col]
    wide = pd.DataFrame({"u": key[unit_col], "r": key[rater_col], "s": cells}).pivot(index="u", columns="r", values="s")
    wide = wide.reindex(index=pd.unique(key[unit_col]), columns=pd.unique(key[rater_col]))
    wide.columns.name = wide.index.name = None
    if wide.isna().to_numpy().any():
        issues.append(("warning", "Bazı birimler her puanlayıcı tarafından puanlanmamış (dengesiz tasarım); bu hücreler eksik sayılır."))
    return _finish(RaterData("wide", wide.astype(object), issues))


def from_counts(df, id_col, category_cols):
    issues = []
    if id_col not in df or not category_cols or any(c not in df for c in category_cols):
        raise RaterDataError("Kategori-sıklık biçiminde birim kimliği ve kategori sütunları seçilmelidir.")
    if id_col in category_cols or len(set(category_cols)) != len(category_cols):
        raise RaterDataError("Kategori sütunları benzersiz olmalı ve birim kimliğini içermemelidir.")
    if df[id_col].isna().any():
        raise RaterDataError("Birim kimliği boş olamaz.")
    counts = df[list(category_cols)]
    if not all(pd.api.types.is_numeric_dtype(counts[c]) for c in counts):
        raise RaterDataError("Kategori-sıklık hücreleri sayı (puanlayıcı sayısı) olmalıdır.")
    v = counts.to_numpy(float)
    if not np.isfinite(v).all() or (v < 0).any() or (v != np.round(v)).any():
        raise RaterDataError("Sıklıklar negatif olmayan tam sayı olmalıdır; eksik hücre olamaz.")
    totals = v.sum(axis=1)
    if len(set(totals)) != 1:
        raise RaterDataError("Her birimi aynı sayıda puanlayıcı sınıflamalıdır (satır toplamları farklı: "
                             + ", ".join(map(str, sorted(set(totals.astype(int))))) + ").")
    if totals[0] < 2:
        raise RaterDataError("Her birim en az iki puanlayıcı tarafından sınıflanmalıdır.")
    dup = df[id_col][df[id_col].duplicated()].astype(str).unique().tolist()
    if dup:
        raise RaterDataError("Tekrarlanan birim kimlikleri: " + ", ".join(dup[:10]))
    table = pd.DataFrame(v.astype(int), index=df[id_col].astype(str).tolist(), columns=list(category_cols))
    return _finish(RaterData("counts", table, issues))


def _finish(data):
    if data.n_units < 2:
        raise RaterDataError("En az iki puanlanan birim gerekir.")
    if data.layout == "wide":
        if data.table.shape[1] < 2:
            raise RaterDataError("En az iki puanlayıcı sütunu gerekir.")
        if data.table.isna().all(axis=1).any():
            data.issues.append(("warning", "Hiç puanı olmayan birimler var."))
        if len(data.categories()) < 2:
            data.issues.append(("warning", "Tüm puanlar aynı (sabit puanlama); şansa göre düzeltilmiş katsayılar tanımsız olabilir."))
    return data


def to_long(data):
    if data.layout != "wide":
        raise RaterDataError("Kategori-sıklık tablosu puanlayıcı kimliği içermediğinden uzun biçime çevrilemez.")
    t = data.table.copy(); t.index.name = "Birim_ID"
    out = t.reset_index().melt(id_vars="Birim_ID", var_name="Puanlayici_ID", value_name="Puan")
    return out.dropna(subset=["Puan"]).reset_index(drop=True)


def counts_from_wide(data):
    if data.layout == "counts":
        return data.table
    if data.has_missing:
        raise RaterDataError("Eksik puan varken kategori-sıklık tablosu oluşturulamaz.")
    cats = data.categories()
    return pd.DataFrame([[int((row == c).sum()) for c in cats] for row in data.table.to_numpy()],
                        index=data.table.index, columns=[_label(c) for c in cats])


def _label(c):
    return str(int(c)) if _is_number(c) and float(c).is_integer() else str(c)


def summary(data):
    t = data.table
    s = {"Biçim": "Geniş (satır = birim, sütun = puanlayıcı)" if data.layout == "wide" else "Kategori-sıklık (satır = birim, sütun = kategori)",
         "Birim sayısı": data.n_units, "Puanlayıcı sayısı": data.n_raters,
         "Kategori sayısı": len(data.categories()), "Kategoriler": ", ".join(map(_label, data.categories()))}
    if data.layout == "wide":
        s["Eksik hücre"] = int(t.isna().to_numpy().sum())
        s["Puan türü"] = "sayısal" if data.numeric else "metin (nominal)"
    return s


def applicable(method, data):
    """(bool, Turkish reason) using the method's actual estimator assumptions."""
    m = METHODS[method]
    errors = [msg for sev, msg in data.issues if sev == "error"]
    if errors:
        return False, "Veri hataları giderilmeden analiz yapılamaz."
    if data.layout == "counts" and method not in COUNTS_OK:
        return False, "Bu yöntem puanlayıcı kimliği gerektirir; kategori-sıklık tablosuyla çalışmaz."
    if data.has_missing and method not in MISSING_OK:
        return False, "Eksik puan var; bu tahminci tam veri gerektirir (yalnız Krippendorff alfa eksik puana izin verir)."
    if method in TWO_RATER and data.n_raters != 2:
        return False, f"Tam olarak iki puanlayıcı gerekir (veride {data.n_raters})."
    cats = data.categories()
    if method in NUMERIC and not data.numeric:
        return False, "Sayısal (sıralı/aralık) puan gerektirir."
    if method in BINARY and len(cats) != 2:
        return False, f"Tam olarak iki kategori gerekir (veride {len(cats)})."
    if method in MIN_CATEGORIES and len(cats) < MIN_CATEGORIES[method]:
        return False, f"En az {MIN_CATEGORIES[method]} kategori gerekir."
    if method == "kripp_ratio" and min(cats) <= 0:
        return False, "Oran düzeyi pozitif puanlar gerektirir."
    if method in {"kripp_nominal", "kripp_ordinal", "kripp_interval", "kripp_ratio"}:
        if int((data.table.notna().sum(axis=1) >= 2).sum()) < 1:
            return False, "En az iki puanı olan birim gerekir."
    return True, ""


def _wide(data):
    return data.table.to_numpy(object)


def _table2(data, cats):
    x = _wide(data)
    idx = {c: i for i, c in enumerate(cats)}
    t = np.zeros((len(cats), len(cats)))
    for a, b in x:
        t[idx[a], idx[b]] += 1
    return t


def _counts(data, cats=None):
    if data.layout == "counts":
        return data.table.to_numpy(float)
    cats = data.categories() if cats is None else cats
    x = _wide(data)
    return np.array([[sum(1 for v in row if v == c) for c in cats] for row in x], float)


def _pa(counts, weights=None):
    r = counts.sum(axis=1)
    star = counts if weights is None else counts @ weights.T
    return float(np.mean((counts * (star - 1)).sum(axis=1) / (r * (r - 1))))


def _quadratic_weights(cats):
    x = np.asarray(cats, float)
    span = x.max() - x.min()
    return 1 - (x[:, None] - x[None, :]) ** 2 / span ** 2


def _ratio(num, den):
    return float(num / den) if den != 0 else float("nan")


def _kappa2(data, wt):
    from statsmodels.stats.inter_rater import cohens_kappa
    cats = data.categories()
    r = cohens_kappa(_table2(data, cats), wt=wt)
    out = {"estimate": float(r.kappa)}
    if np.isfinite(r.kappa) and r.std_kappa0 > 0:
        out.update(se=float(r.std_kappa), ci95=[float(r.kappa_low), float(r.kappa_upp)], z0=float(r.z_value),
                   p=float(r.pvalue_two_sided))
    return out


def _icc(data, kind):
    import pingouin as pg
    long = to_long(data).astype({"Puan": float})
    t = pg.intraclass_corr(long, targets="Birim_ID", raters="Puanlayici_ID", ratings="Puan").set_index("Type")
    row = t.loc[kind]
    return {"estimate": float(row.ICC), "F": float(row.F), "df1": float(row.df1), "df2": float(row.df2),
            "p": float(row.pval), "ci95": _icc_ci(np.asarray(_wide(data), float), kind)}


def _icc_ci(x, kind, alpha=.05):
    """McGraw & Wong (1996) 95% CIs, the same formulas pingouin 0.6 uses but without its 2-decimal rounding."""
    from scipy.stats import f
    n, k = x.shape; g = x.mean(); rows, cols = x.mean(axis=1), x.mean(axis=0)
    ssr, ssc = k * ((rows - g) ** 2).sum(), n * ((cols - g) ** 2).sum(); sse = ((x - g) ** 2).sum() - ssr - ssc
    msb, msj, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1)); msw = (ssc + sse) / (n * (k - 1))
    df1, df1kd, df2kd = n - 1, n * (k - 1), (n - 1) * (k - 1); q = 1 - alpha / 2
    if kind in ("ICC(1,1)", "ICC(1,k)"):
        F = msb / msw; lo, hi = F / f.ppf(q, df1, df1kd), F * f.ppf(q, df1kd, df1)
        return [float(v) for v in ([(lo - 1) / (lo + k - 1), (hi - 1) / (hi + k - 1)] if kind == "ICC(1,1)" else [1 - 1 / lo, 1 - 1 / hi])]
    if kind in ("ICC(C,1)", "ICC(C,k)"):
        F = msb / mse; lo, hi = F / f.ppf(q, df1, df2kd), F * f.ppf(q, df2kd, df1)
        return [float(v) for v in ([(lo - 1) / (lo + k - 1), (hi - 1) / (hi + k - 1)] if kind == "ICC(C,1)" else [1 - 1 / lo, 1 - 1 / hi])]
    # ICC(A,1) and ICC(A,k): McGraw & Wong (1996, Table 7) Satterthwaite form, as in R irr::icc.
    r = (msb - mse) / (msb + (k - 1) * mse + k * (msj - mse) / n)
    rk = (msb - mse) / (msb + (msj - mse) / n)
    rr = r if kind == "ICC(A,1)" else rk
    a = k * rr / (n * (1 - rr)) if kind == "ICC(A,1)" else rr / (n * (1 - rr))
    b = 1 + (k * rr * (n - 1)) / (n * (1 - rr)) if kind == "ICC(A,1)" else 1 + (rr * (n - 1)) / (n * (1 - rr))
    v = (a * msj + b * mse) ** 2 / ((a * msj) ** 2 / (k - 1) + (b * mse) ** 2 / ((n - 1) * (k - 1)))
    fl, fu = f.ppf(q, n - 1, v), f.ppf(q, v, n - 1)
    if kind == "ICC(A,1)":
        out = [n * (msb - fl * mse) / (fl * (k * msj + (k * n - k - n) * mse) + n * msb),
               n * (fu * msb - mse) / (k * msj + (k * n - k - n) * mse + n * fu * msb)]
    else:
        out = [n * (msb - fl * mse) / (fl * (msj - mse) + n * msb), n * (fu * msb - mse) / (msj - mse + n * fu * msb)]
    return [float(v) for v in out]


def _kendall(data):
    x = np.asarray(_wide(data), float)            # rows = ranked objects (n), columns = raters (k)
    n, k = x.shape
    ranks = np.column_stack([stats.rankdata(x[:, j]) for j in range(k)])
    ties = sum(float(((c := np.unique(x[:, j], return_counts=True)[1]) ** 3 - c).sum()) for j in range(k))
    R = ranks.sum(axis=1)
    w = (12 * (R ** 2).sum() - 3 * k ** 2 * n * (n + 1) ** 2) / (k ** 2 * n * (n ** 2 - 1) - k * ties)
    chi2 = k * (n - 1) * w
    return {"estimate": float(w), "chi2": float(chi2), "df": n - 1, "p": float(stats.chi2.sf(chi2, n - 1))}


def _bland_altman(data):
    x = np.asarray(_wide(data), float)
    d = x[:, 0] - x[:, 1]; n = len(d); bias = float(d.mean()); sd = float(d.std(ddof=1))
    t = float(stats.t.ppf(.975, n - 1)); loa = [bias - 1.96 * sd, bias + 1.96 * sd]
    h_bias, h_loa = t * sd / math.sqrt(n), t * sd * math.sqrt(3 / n)
    return {"bias": bias, "sd": sd, "loa": loa, "ci95_bias": [bias - h_bias, bias + h_bias],
            "ci95_loa": [[loa[0] - h_loa, loa[0] + h_loa], [loa[1] - h_loa, loa[1] + h_loa]], "n": n,
            "mean": ((x[:, 0] + x[:, 1]) / 2).tolist(), "difference": d.tolist()}


def compute(method, data):
    ok, reason = applicable(method, data)
    if not ok:
        raise RaterDataError(reason)
    from statsmodels.stats.inter_rater import cohens_kappa, fleiss_kappa
    cats = data.categories()
    if method == "cohen_kappa": return _kappa2(data, None)
    if method == "cohen_kappa_linear": return _kappa2(data, "linear")
    if method == "cohen_kappa_quadratic": return _kappa2(data, "quadratic")
    if method in ("fleiss_kappa", "scott_pi"): return {"estimate": float(fleiss_kappa(_counts(data), method="fleiss"))}
    if method == "brennan_prediger": return {"estimate": float(fleiss_kappa(_counts(data), method="randolph"))}
    if method == "percent_agreement":
        c = _counts(data)
        return {"estimate": _pa(c), "all_agree": float(np.mean(c.max(axis=1) == c.sum(axis=1)))}
    if method == "gwet_ac1":
        c = _counts(data); q = c.shape[1]; pi = (c / c.sum(axis=1, keepdims=True)).mean(axis=0)
        pa, pe = _pa(c), float((pi * (1 - pi)).sum() / (q - 1))
        return {"estimate": _ratio(pa - pe, 1 - pe)}
    if method == "gwet_ac2":
        c = _counts(data, cats); q = len(cats); w = _quadratic_weights(cats)
        pi = (c / c.sum(axis=1, keepdims=True)).mean(axis=0)
        pa, pe = _pa(c, w), float(w.sum() * (pi * (1 - pi)).sum() / (q * (q - 1)))
        return {"estimate": _ratio(pa - pe, 1 - pe)}
    if method == "conger_kappa":
        x = _wide(data); r = x.shape[1]
        p = np.array([[np.mean(x[:, g] == c) for c in cats] for g in range(r)])     # raters x categories
        pe = float((p.mean(axis=0) ** 2 - p.var(axis=0, ddof=1) / r).sum())
        pa = _pa(_counts(data, cats))
        return {"estimate": _ratio(pa - pe, 1 - pe)}
    if method == "light_kappa":
        x = data.table; pairs = list(itertools.combinations(range(x.shape[1]), 2)); vals = []
        for a, b in pairs:
            sub = RaterData("wide", x.iloc[:, [a, b]])
            vals.append(float(cohens_kappa(_table2(sub, cats)).kappa))
        return {"estimate": float(np.mean(vals)), "pairs": len(pairs)}
    if method.startswith("kripp_"):
        import krippendorff
        level = method.split("_")[1]
        values = np.array([[np.nan if _isna(v) else cats.index(v) if not data.numeric else float(v)
                            for v in data.table[c]] for c in data.table.columns], float)
        domain = list(range(len(cats))) if not data.numeric else [float(c) for c in cats]
        return {"estimate": float(krippendorff.alpha(reliability_data=values, level_of_measurement=level, value_domain=domain)),
                "pairable_units": int((data.table.notna().sum(axis=1) >= 2).sum())}
    if method == "pabak":
        t = _table2(data, cats); return {"estimate": float(2 * np.trace(t) / t.sum() - 1)}
    if method == "bangdiwala_b":
        t = _table2(data, cats)
        return {"estimate": _ratio((np.diag(t) ** 2).sum(), (t.sum(axis=1) * t.sum(axis=0)).sum())}
    if method.startswith("icc_"):
        # pingouin 0.6 uses McGraw–Wong labels: A = absolute agreement (Shrout–Fleiss 2), C = consistency (3).
        return _icc(data, {"icc_1_1": "ICC(1,1)", "icc_1_k": "ICC(1,k)", "icc_2_1": "ICC(A,1)", "icc_2_k": "ICC(A,k)",
                           "icc_3_1": "ICC(C,1)", "icc_3_k": "ICC(C,k)"}[method])
    if method == "kendall_w": return _kendall(data)
    if method in ("spearman", "pearson"):
        x = np.asarray(_wide(data), float)
        if method == "spearman":
            r = stats.spearmanr(x[:, 0], x[:, 1]); return {"estimate": float(r.statistic), "p": float(r.pvalue)}
        r = stats.pearsonr(x[:, 0], x[:, 1]); ci = r.confidence_interval(.95)
        return {"estimate": float(r.statistic), "p": float(r.pvalue), "ci95": [float(ci.low), float(ci.high)]}
    if method == "mcnemar":
        from statsmodels.stats.contingency_tables import mcnemar
        r = mcnemar(_table2(data, cats), exact=False, correction=True)
        return {"chi2": float(r.statistic), "df": 1, "p": float(r.pvalue)}
    if method in ("stuart_maxwell", "bhapkar"):
        from statsmodels.stats.contingency_tables import SquareTable
        t = _table2(data, cats); k = len(cats)
        S = -(t + t.T)[:k - 1, :k - 1]; np.fill_diagonal(S, (t.sum(0) + t.sum(1) - 2 * np.diag(t))[:k - 1])
        if np.linalg.matrix_rank(S) < k - 1:
            raise RaterDataError("Marjinal homojenlik kovaryans matrisi tekil (ör. bir kategoride hiç uyuşmazlık yok); "
                                 "test bu tabloda tanımsızdır.")
        r = SquareTable(t, shift_zeros=False).homogeneity(method=method)
        return {"chi2": float(r.statistic), "df": int(r.df), "p": float(r.pvalue)}
    if method == "bland_altman": return _bland_altman(data)
    raise KeyError(method)


EVIDENCE_PATH = Path(__file__).resolve().parents[1] / "validation/rater_reliability/evidence.json"


def evidence():
    try:
        return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))["status"]
    except (OSError, KeyError, ValueError):
        return {}


def status(method):
    """Method-level evidence label from the committed validation record (never 'CERTIFIED')."""
    return evidence().get(method, {}).get("label", "NOT CERTIFIED")


def validated_outputs(method):
    return set(evidence().get(method, {}).get("validated_outputs", []))


def interpret(method, result, data):
    """Short Turkish interpretation built only from VALIDATED displayed values (no universal thresholds)."""
    m = METHODS[method]
    need = ("chi2", "df", "p") if m["group"] == "sistematik" and method != "bland_altman" else \
        ("bias", "loa") if method == "bland_altman" else ("estimate",)
    if any(k not in result for k in need):
        return f"{m['name']}: bu çıktı için bağımsız sayısal kanıt bulunmadığından değer raporlanmamıştır."
    f = lambda v: f"{v:.3f}"
    if m["group"] == "sistematik" and method != "bland_altman":
        sig = "istatistiksel olarak anlamlı" if result["p"] < .05 else "α = .05 düzeyinde anlamlı değil"
        return (f"{m['name']}: χ²({result['df']}) = {f(result['chi2'])}, p = {result['p']:.4f}. Puanlayıcıların marjinal "
                f"dağılımları arasındaki fark {sig}; bu test uyumun derecesini değil sistematik farklılığı (yanlılığı) sınar.")
    if method == "bland_altman":
        return (f"Ortalama fark (yanlılık) {f(result['bias'])}, %95 uyum sınırları [{f(result['loa'][0])}, {f(result['loa'][1])}]. "
                "Sınırların kabul edilebilirliği ölçme bağlamında önceden belirlenen klinik/uygulamalı ölçüte göre değerlendirilmelidir.")
    est = result["estimate"]
    if not np.isfinite(est):
        return f"{m['name']} bu veride tanımsızdır (ör. sabit puanlama nedeniyle şans uyumu 1)."
    kind = {"spearman": "ilişki (birlikte değişme) ölçüsüdür; uyumu değil", "pearson": "doğrusal ilişki ölçüsüdür; mutlak uyumu değil",
            "kendall_w": "puanlayıcılar arası sıralama uyumunu"}.get(method, "şansa göre düzeltilmiş uyumu" if "kappa" in method or method in
            {"scott_pi", "gwet_ac1", "gwet_ac2", "brennan_prediger", "pabak"} or method.startswith("kripp") else "uyum/güvenirliği")
    text = f"{m['name']} ({m['abbr']}) = {f(est)}; {data.n_units} birim ve {data.n_raters} puanlayıcı. Bu değer {kind} yansıtır."
    if "ci95" in result:
        text += f" %95 GA [{f(result['ci95'][0])}, {f(result['ci95'][1])}]."
    text += (" Yorum için evrensel eşik kullanılmamıştır; değer, kategori sayısı, prevalans ve çalışmanın amacıyla birlikte "
             "değerlendirilmelidir.")
    if method.startswith("icc_"):
        text += " " + m["formula"].split(":")[0] + " modeli varsayılmıştır."
    return text


def versions():
    out = {}
    for p in ("numpy", "pandas", "scipy", "statsmodels", "pingouin", "krippendorff"):
        try:
            out[p] = version(p)
        except Exception:
            out[p] = "yok"
    return out


def run(method, data):
    result = compute(method, data)
    keep = validated_outputs(method)
    # Evidence is output-specific: an unvalidated statistic must not leak via
    # the generic result display, even when it is a primary coefficient.
    # Metadata and plotting coordinates are not inferential results.
    metadata = {"n", "pairs", "pairable_units", "mean", "difference"}
    shown = {k: v for k, v in result.items() if k in keep or k in metadata}
    return {"method": method, "result": result, "shown": shown, "status": status(method),
            "interpretation": interpret(method, shown, data), "n_units": data.n_units, "n_raters": data.n_raters,
            "categories": [_label(c) for c in data.categories()], "engine": METHODS[method]["engine"], "versions": versions()}
