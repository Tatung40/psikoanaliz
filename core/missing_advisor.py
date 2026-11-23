"""Central deterministic missing-data advisor (no LLM/API). Recommends only methods the platform supports, never
diagnoses MCAR/MAR/MNAR, never imputes or deletes by itself, and never offers multiple imputation (no pooling).
"""
import pandas as pd

# Engine support per analysis route (prefix match on the page name). native=True: the estimator uses the observed
# responses directly (no imputation needed); False: the engine requires a complete matrix (verified refusal);
# None: not verified for this route.
_MIRT = dict(native=False, estimator="Python mirt marjinal ML (EM)", data="binary/polytomous",
             note="Python mirt adaptörü eksik yanıtı reddeder (doğrulanmış eksik veri politikası yok).")
ENGINES = tuple((p, _MIRT) for p in ("MTK: 1PL", "MTK: 2PL", "MTK: 3PL", "MTK: 4PL", "MTK: Karşılaştırma", "MTK: Python mirt")) + (
    ("MTK: RaschPy MFRM", dict(native=True, estimator="RaschPy RCL (ikili koşullu olabilirlik)", data="long",
                               note="Eksik puan satırları gözlenen puanlarla kestirilir ve raporlanır; atama gerekmez.")),
    ("MTK: Çok Yüzeyli Rasch", dict(native=True, estimator="R/TAM marjinal ML", data="long",
                                    note="Olabilirlik yalnız gözlenen puanlarla kurulur; eksik satırlar raporlanır.")),
)
REFERENCES = ("Little & Rubin (2019), Statistical Analysis with Missing Data (3rd ed.).",
              "Enders (2010), Applied Missing Data Analysis.",
              "Schafer & Graham (2002), Psychological Methods, 7(2), 147-177.")
LISTWISE_MAX_ROW_LOSS = .05                              # complete-case analysis only when it loses <= 5% of rows
HIGH_ITEM_RATE = .30


def engine(page):
    for prefix, rec in ENGINES:
        if str(page).startswith(prefix):
            return rec
    return dict(native=None, estimator="doğrulanmadı", data="any",
                note="Bu analiz motorunun eksik veri desteği platformda doğrulanmamıştır.")


def data_type(x):
    vals = pd.unique(x.stack().dropna())
    if set(vals) <= {0, 1}:
        return "binary"
    if all(float(v).is_integer() for v in vals) and len(vals) <= 7:
        return "ordinal"
    return "continuous"


def advise(frame, columns, page):
    """Deterministic recommendation. Returns status, recommended method (a key of core.missing_imputation.METHODS or
    None), allowed methods with reasons, excluded methods with reasons, a pattern summary and references."""
    if not columns:
        raise ValueError("Madde sütunu seçilmedi.")
    x = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    miss = x.isna(); n, k = x.shape; cells = int(miss.to_numpy().sum())
    eng = engine(page); dtype = data_type(x)
    rows_any = int(miss.any(axis=1).sum()); rates = miss.mean()
    pattern = dict(n=n, k=k, missing_cells=cells, missing_pct=100 * cells / (n * k) if n * k else 0., rows_with_missing=rows_any,
                   row_loss_pct=100 * rows_any / n if n else 0., complete_rows=n - rows_any,
                   high_missing_items=[str(c) for c in rates[rates > HIGH_ITEM_RATE].index],
                   all_missing_items=[str(c) for c in rates[rates == 1].index], data_type=dtype)
    out = dict(page=page, columns=list(map(str, columns)), engine=eng, pattern=pattern, references=list(REFERENCES),
               mechanism_note="Kayıp mekanizması (MCAR/MAR/MNAR) gözlenen veriden kesin olarak belirlenemez; öneri "
                              "mekanizma varsayımına duyarlıdır ve duyarlılık analizi önerilir.",
               multiple_imputation="Çoklu atama sunulmaz: bu analizde atanmış veri setlerinin sonuçlarını birleştirme "
                                   "(Rubin kuralları) desteği yoktur.")
    if cells == 0:
        return dict(out, status="none", recommended=None, allowed={}, excluded={},
                    reason="Seçilen sütunlarda kayıp veri yok; işlem gerekmez.")
    if eng["native"] is True:
        return dict(out, status="native", recommended="keep", allowed={"keep": eng["note"]}, excluded={},
                    reason=f"{eng['estimator']} eksik gözlemleri doğrudan işler; atamaya zorlanmaz. {eng['note']}")
    allowed, excluded = {}, {}
    small_loss = pattern["row_loss_pct"] <= 100 * LISTWISE_MAX_ROW_LOSS
    allowed["listwise"] = ("Tam kayıt analizi: MCAR altında yansız, ancak güç kaybı ve MCAR dışında seçilim yanlılığı riski "
                           f"vardır (kayıp satır %{pattern['row_loss_pct']:.1f}).")
    if dtype in ("binary", "ordinal"):
        allowed["mode"] = "En sık kategori: kategori yapısını korur; varyansı ve standart hataları küçültür (tek atama)."
        allowed["knn"] = ("KNN (yuvarlamalı): benzer yanıt örüntülerinden atar; " +
                          ("ikili maddelerde 0,5 eşiğiyle yuvarlama onayı gerekir; " if dtype == "binary" else "") +
                          "tek atamadır, belirsizliği küçük gösterir.")
        excluded["median"] = "Medyan sıralı/ikili kategorilerde kesirli değer üretebilir; kategori yapısını korumaz."
        if dtype == "ordinal":
            excluded["knn"] = "KNN sıralı kategorilerde kesirli değer üretebilir ve yuvarlama yalnız ikili maddede desteklenir."
            allowed.pop("knn")
    else:
        allowed["median"] = "Medyan: sürekli puanlarda basit tek atama; varyansı küçültür."
        allowed["knn"] = "KNN: sürekli puanlarda örüntü tabanlı tek atama; belirsizliği küçük gösterir."
        excluded["mode"] = "En sık değer sürekli puanlar için uygun değildir."
    if dtype == "binary":
        allowed["zero"] = ("0 puan: YALNIZ puanlama kuralınız atlanan yanıtı yanlış sayıyorsa (araştırmacı kararı). Bu bir "
                           "istatistiksel atama değil puanlama kararıdır; otomatik önerilmez.")
    else:
        excluded["zero"] = "0 ile doldurma yalnız ikili başarı testlerinde puanlama kuralı olarak savunulabilir."
    excluded["keep"] = f"{eng['estimator']} eksik yanıtla çalışmaz." if eng["native"] is False else \
        "Bu motorun eksik veri desteği doğrulanmadı; kayıpları koruyarak çalıştırma önerilmez."
    if pattern["all_missing_items"]:
        excluded.update({m: "Tamamen boş madde(ler) var; önce bu maddeleri analizden çıkarın: " +
                         ", ".join(pattern["all_missing_items"]) for m in ("mode", "knn", "median") if m in allowed})
        for m in ("mode", "knn", "median"): allowed.pop(m, None)
    rec = "listwise" if small_loss else ("mode" if dtype in ("binary", "ordinal") else "knn")
    if rec not in allowed: rec = "listwise"
    reason = (f"{eng['estimator']} eksiksiz matris gerektirir ({eng['note']}). " if eng["native"] is False else
              "Bu motorun eksik veri desteği doğrulanmadı. ") + \
             (f"Kayıp satır oranı %{pattern['row_loss_pct']:.1f} (≤ %{100 * LISTWISE_MAX_ROW_LOSS:.0f}) olduğundan tam kayıt "
              "analizi önerilir; tek atamanın belirsizliği küçültme riski yoktur." if small_loss else
              f"Kayıp satır oranı %{pattern['row_loss_pct']:.1f} (> %{100 * LISTWISE_MAX_ROW_LOSS:.0f}); tam kayıt analizi çok "
              f"gözlem kaybettirir. Kategori yapısını koruyan '{rec}' tek ataması önerilir ve tam kayıt analizi ile duyarlılık "
              "karşılaştırması yapılmalıdır.")
    if pattern["high_missing_items"]:
        reason += f" %{100 * HIGH_ITEM_RATE:.0f}'dan fazla kayıp: {', '.join(pattern['high_missing_items'])} (madde düzeyinde inceleyin)."
    return dict(out, status="action", recommended=rec, allowed=allowed, excluded=excluded, reason=reason)


def method_label(method):
    return {"keep": "Kayıpları koru (motor doğrudan işler)", "listwise": "Tam kayıt analizi (eksik satırları çıkar)",
            "zero": "0 puan (puanlama kuralı)", "mode": "En sık kategori ile tek atama", "median": "Medyan ile tek atama",
            "knn": "KNN ile tek atama"}.get(method, str(method))
