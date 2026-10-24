"""Evidence-based usage limits for 3PL/4PL parameter interpretation (Python mirt 1.2.0).

Derived ONLY from the pre-registered study validation/irt_3pl4pl_root_cause.py
(evidence: validation/evidence/p1_3pl4pl/irt_3pl4pl_root_cause.json; criteria fixed before results:
RMSE a/b <= .30, c/d <= .07, IRF RMSE <= .02, <= 5 % parameters at a box bound; theta ~ N(0, 1);
k in {12, 30}; N in {1000, 3000, 10000}; 20 replications per cell).

Findings that the rules encode:
  * ML: optimization is not the problem (21 vs 61 nodes, tolerance and true-value starts give the same
    maximum; logL(estimate) > logL(truth) in every fit). Every ML fit put parameters on the library box
    (a = 5, c = 0, d = 1): the likelihood is flat along a-c-d ridges, so parameters are weakly identified
    even though item response functions are recovered well.
  * 3PL ML met the criteria only at N = 10000; 4PL ML never (a RMSE .24-1.86).
  * MAP with the library default priors met the criteria for 3PL and 4PL at N >= 3000 (k = 12 and 30)
    with no boundary estimates; at N = 1000 the IRF error criterion failed.
  * A skewed latent distribution (standardized chi-square(4)) produced a RMSE .41-.82 even with MAP.
Conditions outside the studied design (N < 1000, k < 12, non-normal theta) are reported as not studied.
"""
EVIDENCE = "validation/evidence/p1_3pl4pl/irt_3pl4pl_root_cause.json"
_MIN_N_SUPPORTED = {("3PL", "ML"): 10000, ("3PL", "MAP"): 3000, ("4PL", "ML"): None, ("4PL", "MAP"): 3000}


def usage_status(model, estimation, n, k):
    """Return (status, Turkish message). status in {"supported", "caution", "not_supported", "not_applicable"}."""
    if model not in ("3PL", "4PL"):
        return "not_applicable", ""
    key = (model, estimation)
    if key not in _MIN_N_SUPPORTED:
        raise ValueError("Bilinmeyen kestirim yöntemi.")
    needed = _MIN_N_SUPPORTED[key]
    label = "ML (önselsiz)" if estimation == "ML" else "MAP (mirt varsayılan önselleri)"
    tail = (f" Kanıt: {EVIDENCE} (önceden tanımlanmış ölçütler; θ ~ N(0,1) varsayımı). Normal olmayan θ "
            "dağılımında hata belirgin biçimde artar.")
    if k < 12:
        return "not_supported", (f"{model} {label}: {k} madde incelenen tasarımın (k ≥ 12) dışında; parametreler "
                                 "yorumlanmamalı." + tail)
    if needed is None:
        return "not_supported", (f"{model} {label}: hiçbir incelenen örneklem büyüklüğünde (N ≤ 10000) parametre "
                                 "kurtarma ölçütleri sağlanmadı; ML kestirimleri sistematik olarak kutu sınırlarına "
                                 "(a = 5, c = 0, d = 1) dayandı. MAP seçeneğini kullanın." + tail)
    if n >= needed:
        return "supported", (f"{model} {label}: N = {n} ≥ {needed} için önceden tanımlanmış kurtarma ölçütleri "
                             "karşılandı." + tail)
    if n >= 1000:
        return "caution", (f"{model} {label}: N = {n} < {needed}. Madde tepki fonksiyonları yaklaşık olarak "
                           "kurtarılabilir ancak a/b/c(/d) parametreleri tek tek yorumlanmamalıdır." + tail)
    return "not_supported", (f"{model} {label}: N = {n} < 1000 incelenen tasarımın dışında; parametre "
                             "kurtarma güvencesi yok." + tail)
