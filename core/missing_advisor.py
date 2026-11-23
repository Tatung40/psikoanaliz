"""Central deterministic missing-data advisor (no LLM/API). Recommends only methods the platform supports, never
diagnoses MCAR/MAR/MNAR, never imputes or deletes by itself, and never offers multiple imputation (no pooling).
"""
import pandas as pd

# Engine support per analysis route (prefix match on the page name). native=True: the estimator uses the observed
# responses directly (no imputation needed); False: the engine requires a complete matrix (verified refusal);
# None: not verified for this route.
_MIRT = dict(native=False, estimator="Python mirt marjinal ML (EM)", data="binary/polytomous",
             note="Python mirt adaptörü eksik yanıtı reddeder (doğrulanmış eksik veri politikası yok).")
_TOTAL = dict(native=False, estimator="KTK toplam puan motoru (core.scoring.calculate_total_scores)", data="any",
              note="Motorun kendi kuralı (doğrulandı): maddelerin en az %80'i yanıtlanmışsa yanıtlanan maddelerin toplamı alınır "
                   "(eksik yanıt toplamda 0 gibi etki eder), daha az yanıtta toplam puan hesaplanmaz.")
_REL = dict(native=False, estimator="Güvenirlik motoru (core.reliability.compute_reliability_indices)", data="any",
            note="Motorun kendi kuralı (doğrulandı): eksik yanıtı olan satırları dışlar (tam kayıt); kullanılan N raporlanır.")
ENGINES = tuple((p, _TOTAL) for p in ("4. Test İstatistikleri", "5. Madde Analizi", "6. Çeldirici Analizi",
                                      "8. Grup Bazlı Analizler", "9. Bilimsel Grafikler", "10. Raporlama")) + \
    (("7. Güvenirlik", _REL),) + tuple((p, _MIRT) for p in ("MTK: 1PL", "MTK: 2PL", "MTK: 3PL", "MTK: 4PL", "MTK: Karşılaştırma", "MTK: Python mirt")) + (
    ("MTK: RaschPy MFRM", dict(native=True, estimator="RaschPy RCL (ikili koşullu olabilirlik)", data="long",
                               note="Eksik puan satırları gözlenen puanlarla kestirilir ve raporlanır; atama gerekmez.")),
    ("MTK: Çok Yüzeyli Rasch", dict(native=True, estimator="R/TAM marjinal ML", data="long",
                                    note="Olabilirlik yalnız gözlenen puanlarla kurulur; eksik satırlar raporlanır.")),
)
REFERENCES = ("Little & Rubin (2019), Statistical Analysis with Missing Data (3rd ed.).",
              "Enders (2010), Applied Missing Data Analysis.",
              "Schafer & Graham (2002), Psychological Methods, 7(2), 147-177.")
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
    sens = "Duyarlılık analizi seçeneği — potansiyel olarak yanlı (tek atama): "
    allowed["listwise"] = ("Tam kayıt analizi: yalnız MCAR varsayımı altında yansızdır; MAR/MNAR altında yanlı olabilir ve "
                           f"güç kaybettirir (kayıp satır %{pattern['row_loss_pct']:.1f}). Düşük kayıp oranı tek başına gerekçe değildir.")
    if dtype in ("binary", "ordinal"):
        allowed["mode"] = sens + "en sık kategori; kategori yapısını korur ama dağılımı tek kategoriye yığar, ilişkileri zayıflatır ve standart hataları küçültür."
        allowed["knn"] = sens + ("KNN; ikili maddelerde 0,5 eşiğiyle yuvarlama onayı gerekir; " if dtype == "binary" else "KNN; ") + \
            "belirsizliği küçük gösterir."
        excluded["median"] = "Medyan sıralı/ikili kategorilerde kesirli değer üretebilir; kategori yapısını korumaz."
        if dtype == "ordinal":
            excluded["knn"] = "KNN sıralı kategorilerde kesirli değer üretebilir ve yuvarlama yalnız ikili maddede desteklenir."
            allowed.pop("knn")
    else:
        allowed["median"] = sens + "medyan; varyansı ve ilişkileri küçültür."
        allowed["knn"] = sens + "KNN; belirsizliği küçük gösterir."
        excluded["mode"] = "En sık değer sürekli puanlar için uygun değildir."
    if dtype == "binary":
        allowed["zero"] = ("0 puan: YALNIZ puanlama kuralınız atlanan yanıtı yanlış sayıyorsa (araştırmacı kararı). Bu bir "
                           "istatistiksel atama değil puanlama kararıdır; otomatik önerilmez.")
    else:
        excluded["zero"] = "0 ile doldurma yalnız ikili başarı testlerinde puanlama kuralı olarak savunulabilir."
    excluded["keep"] = f"{eng['estimator']} eksik yanıtla doğrudan (olabilirlik temelli) çalışmaz." if eng["native"] is False else \
        "Bu motorun eksik veri desteği doğrulanmadı; kayıpları koruyarak çalıştırma önerilmez."
    if pattern["all_missing_items"]:
        excluded.update({m: "Tamamen boş madde(ler) var; önce bu maddeleri analizden çıkarın: " +
                         ", ".join(pattern["all_missing_items"]) for m in ("mode", "knn", "median") if m in allowed})
        for m in ("mode", "knn", "median"): allowed.pop(m, None)
    reason = (f"{eng['estimator']}: {eng['note']} " if eng["native"] is False else "Bu motorun eksik veri desteği doğrulanmadı. ") + \
             ("Bu analiz için platformda bilimsel olarak savunulabilir birincil yöntem (olabilirlik temelli/FIML veya birleştirmeli "
              "çoklu atama) desteklenmiyor; bu nedenle önerilen birincil yöntem yoktur. Aşağıdaki seçenekler varsayımlarıyla "
              "birlikte sunulur: tam kayıt analizi yalnız MCAR altında yansızdır; atama seçenekleri yalnız duyarlılık analizi içindir.")
    if pattern["high_missing_items"]:
        reason += f" %{100 * HIGH_ITEM_RATE:.0f}'dan fazla kayıp: {', '.join(pattern['high_missing_items'])} (madde düzeyinde inceleyin)."
    return dict(out, status="action", recommended=None, allowed=allowed, excluded=excluded, reason=reason)


def method_label(method):
    return {"keep": "Kayıpları koru (motor doğrudan işler)", "listwise": "Tam kayıt analizi (eksik satırları çıkar)",
            "zero": "0 puan (puanlama kuralı)", "mode": "En sık kategori ile tek atama", "median": "Medyan ile tek atama",
            "knn": "KNN ile tek atama"}.get(method, str(method))
