"""Evidence-informed decision aid: absence of missingness never requires imputation.
This does not infer MCAR, MAR or MNAR from observed data.
"""
def assess_necessity(frame, columns, *, analysis="IRT", supports_missing=False):
    if not columns:
        raise ValueError("Select item columns.")
    missing = int(frame.loc[:, columns].isna().sum().sum())
    if missing == 0:
        return {"status": "not_needed", "needed": False,
                "reason": "Secilen maddelerde kayip veri yok; atama gereksiz.",
                "references": ["Little & Rubin (2019), Statistical Analysis with Missing Data, 3rd ed."]}
    if supports_missing:
        return {"status": "not_required", "needed": False,
                "reason": "Secilen analiz eksik gozlemleri dogrudan isleyebiliyorsa tek atama zorunlu degildir. Varsayimlar ve duyarlilik analizi yine incelenmelidir.",
                "references": ["Enders (2010), Applied Missing Data Analysis.",
                               "Little & Rubin (2019), Statistical Analysis with Missing Data."]}
    return {"status": "review", "needed": None,
            "reason": f"{analysis} motoru eksiksiz matris istiyor. Bu durum atamanin otomatik olarak en iyi cozum oldugunu gostermez. Eksik veri destekli motor veya duyarlilik analizi degerlendirilmelidir.",
            "references": ["Schafer & Graham (2002), Psychological Methods, 7(2), 147-177.",
                           "Enders (2010), Applied Missing Data Analysis."]}
