"""3PL/4PL conditional EAP from the supported Python mirt scoring API.

No recalibration, finite plotting-grid substitution, or inference certification.
The posterior SD conditions on fitted items; it is not a frequentist SE/CI.
"""
import numpy as np
import pandas as pd

SCORING_VERSION = "binary34-gh121-241-v1"
QUADRATURE_TOL = 1e-6


def binary_person_abilities(outcome, responses, identifiers=None):
    import mirt
    if outcome.model not in ("3PL", "4PL"):
        raise ValueError("Bu yol yalnız 3PL/4PL içindir; diğer modellerin yolu değiştirilmedi.")
    fit = outcome.fit
    if not bool(getattr(fit, "converged", False)):
        raise ValueError("Yakınsamayan kalibrasyondan kişi kestirimi üretilmez.")
    x = np.asarray(responses, float)
    if x.ndim != 2 or x.shape[1] != outcome.n_items or not len(x) or not np.isfinite(x).all() or not np.isin(x, [0, 1]).all():
        raise ValueError("3PL/4PL için aynı madde sırasıyla sonlu 0/1 yanıtları gerekir.")
    for name, expected in (("latent_mean", [0.]), ("latent_covariance", [[1.]])):
        value = getattr(fit, name, None)
        if value is not None and not np.array_equal(np.asarray(value, float), expected):
            raise ValueError("Yalnız tek boyutlu N(0,1) latent ölçek doğrulandı.")
    if getattr(getattr(fit, "model", None), "n_factors", 1) != 1:
        raise ValueError("Çok boyutlu kişi kestirimi bu yolda desteklenmiyor.")
    ids = pd.Series(list(identifiers) if identifiers is not None else np.arange(1, len(x)+1))
    if len(ids) != len(x) or ids.isna().any() or ids.duplicated().any():
        raise ValueError("Kişi kimlikleri aynı satır sırasıyla eksiksiz ve benzersiz olmalıdır.")
    results = []
    for nodes in (121, 241):
        scored = mirt.fscores(fit, x.astype(int), method="EAP", n_quadpts=nodes, person_ids=ids.tolist())
        theta = np.asarray(scored.theta, float).reshape(-1)
        sd = np.asarray(scored.standard_error, float).reshape(-1)
        if theta.shape != (len(x),) or sd.shape != theta.shape or not np.array_equal(scored.person_ids, ids):
            raise ValueError("Kütüphane kişi sırası/boyutu uyuşmuyor.")
        if not np.isfinite(np.r_[theta, sd]).all() or np.any(sd <= 0):
            raise ValueError("Sonlu EAP ve pozitif koşullu posterior SS gerekir.")
        results.append((theta, sd))
    delta = max(float(np.max(abs(a-b))) for a,b in zip(results[0], results[1]))
    if delta > QUADRATURE_TOL:
        raise ValueError(f"Kişi quadrature denetimi geçmedi: GH121/GH241 farkı {delta:.3g}; çıktı kapalı, otomatik tekrar yok.")
    theta, sd = results[-1]
    return pd.DataFrame({
        "Birey": ids, "Ham puan": x.sum(axis=1).astype(int),
        "Yetenek (θ)": theta, "SH (θ)": sd, "Posterior SS (θ)": sd,
        "Yöntem": "EAP (mirt.fscores GH241; GH121 denetimli)",
        "Model": outcome.model,
        "Belirsizlik türü": "Madde kestirimlerine koşullu posterior SS; örnekleme SH/GA değil",
        "Bilimsel durum": "NOT CERTIFIED — kişi geri kazanımı/kapsaması doğrulanmadı",
        "Quadrature max fark": delta,
    })
