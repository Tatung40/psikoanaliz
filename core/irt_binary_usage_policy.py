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
CERTIFICATION = "validation/evidence/binary_certification/CERTIFICATION.json"
_ESTIMATION = {"1PL": "ML", "2PL": "ML", "3PL": "MAP", "4PL": "MAP"}
_OUTPUTS = {"1PL": "madde parametreleri, MTF/bilgi, SH ve %95 Wald GA", "2PL": "madde parametreleri, MTF/bilgi, SH ve %95 Wald GA",
            "3PL": "madde parametreleri, MTF/bilgi, SH ve %95 Wald GA", "4PL": "madde parametreleri ve MTF/bilgi (SH/GA kapsam dışı)"}


def _certification_record():
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / CERTIFICATION
    return json.loads(path.read_text()) if path.exists() else {"models": {}}


def certification(model, estimation, n, k, solver_settings=None):
    """Return (status, Turkish text) from the pre-registered decision (validation/binary_certification).

    status: "certified" (inside the certified rectangle with the certified estimation and solver precision),
    "outside_scope" or "not_certified". Rasch is the fixed-a 1PL; both 1PL variants share the 1PL decision."""
    name = "1PL" if model in ("Rasch", "1PL") else model
    entry = _certification_record()["models"].get(name)
    if entry is None:
        return "not_certified", f"{name}: sertifika kaydı yok (NOT CERTIFIED)."
    if entry["status"] != "CERTIFIED":
        return "not_certified", (f"{name}: NOT CERTIFIED — başarısız kapı: {', '.join(entry['failed_gates'])} "
                                 f"({CERTIFICATION}).")
    scope = entry["scope"]; (n_lo, n_hi), (k_lo, k_hi) = scope["N"], scope["k"]
    settings = solver_settings or {}
    precise = settings.get("n_quadpts", 0) >= 61 and settings.get("tol", 1) <= 1e-4 and settings.get("max_iter", 0) >= 500
    rect = f"{_ESTIMATION[name]}, N {n_lo}–{n_hi}, k {k_lo}–{k_hi}, θ ~ N(0, 1), ≥ 61 GH düğümü"
    if estimation != _ESTIMATION[name] or not (n_lo <= n <= n_hi and k_lo <= k <= k_hi) or not precise:
        return "outside_scope", (f"{name}: bu analiz (kestirim {estimation}, N = {n}, k = {k}, düğüm "
                                 f"{settings.get('n_quadpts', '?')}) sertifika kapsamı dışında — NOT VERIFIED. "
                                 f"Sertifika kapsamı: {rect}.")
    return "certified", (f"CERTIFIED — {name} {rect}. Sertifikalı çıktılar: {_OUTPUTS[name]}. Kanıt: R mirt 1.48 eşdeğerliği "
                         f"ve önceden kayıtlı Monte Carlo ({CERTIFICATION}). θ normalliği veriden doğrulanmaz; "
                         "birey kestirimleri, uyum istatistikleri ve DIF bu sertifikaya dahil değildir.")
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
    if n >= needed and (n not in (1000, 3000, 10000) or k not in (12, 30)):
        return "caution", (f"{model} {label}: N = {n}, k = {k} doğrudan incelenmiş bir tasarım hücresi değil; "
                           "komşu örneklem/madde sayılarındaki kanıt bu tasarımı doğrulamaz." + tail)
    if n >= needed:
        return "supported", (f"{model} {label}: N = {n} ≥ {needed} için önceden tanımlanmış kurtarma ölçütleri "
                             "karşılandı." + tail)
    if n >= 1000:
        return "caution", (f"{model} {label}: N = {n} < {needed}. Madde tepki fonksiyonları yaklaşık olarak "
                           "kurtarılabilir ancak a/b/c(/d) parametreleri tek tek yorumlanmamalıdır." + tail)
    return "not_supported", (f"{model} {label}: N = {n} < 1000 incelenen tasarımın dışında; parametre "
                             "kurtarma güvencesi yok." + tail)


def evidence_note(model, common_discrimination=False):
    """Return (level, Turkish text): the certification decision for this binary library model (pre-fit banner)."""
    name = "1PL" if model in ("Rasch", "1PL") else model
    entry = _certification_record()["models"].get(name)
    if entry is None:
        return "warning", "Bu model için bağımsız doğrulama kaydı yok (NOT VERIFIED)."
    scope = entry["scope"]; (n_lo, n_hi), (k_lo, k_hi) = scope["N"], scope["k"]
    rect = f"{_ESTIMATION[name]}, N {n_lo}–{n_hi}, k {k_lo}–{k_hi}, θ ~ N(0, 1), ≥ 61 GH düğümü"
    if entry["status"] == "CERTIFIED":
        variant = (" (sabit a = 1 ve ortak serbest a)" if name == "1PL" else "")
        return "info", (f"{name}{variant}: CERTIFIED — kapsam {rect}; çıktılar: {_OUTPUTS[name]}. R mirt 1.48 ile "
                        f"bağımsız eşdeğerlik ve önceden kayıtlı Monte Carlo ({CERTIFICATION}). Kapsam dışı "
                        "tasarımlar sonuç ekranında ayrıca işaretlenir.")
    return "warning", (f"{name}: NOT CERTIFIED — başarısız kapı: {', '.join(entry['failed_gates'])} "
                       f"({CERTIFICATION}). Parametre yorumu kullanım sınırlarına bağlıdır.")
