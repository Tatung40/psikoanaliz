"""Item-parameter standard errors and Wald intervals for Python mirt binary fits (1PL–4PL, Rasch).

Reads the library's own covariance (fit.vcov, Oakes observed information); nothing is re-estimated.
The gates of the removed B13 report are kept: no SE without convergence, structural identifiability
(free parameters <= 2^k - 1 response patterns), a finite positive-definite covariance with condition
number <= 1e8, and no Wald interval for a parameter on the library's box bound (the asymptotic normal
reference is invalid there). Fixed parameters (a = 1 in 1PL/Rasch) have no SE.
"""
import numpy as np
import pandas as pd

Z95 = 1.959963984540054
NAMES = {"discrimination": "a", "difficulty": "b", "guessing": "c", "upper": "d"}
STATUS = "EXPERIMENTAL: Oakes gözlenen bilgi SH; Wald GA bağımsız kapsama çalışmasıyla sertifikalı değildir."


def _matrix(value):
    return np.asarray(value() if callable(value) else value, float)


def parameter_inference(fit, model, item_names):
    """Return (table, reason). table is None and reason explains why when SE cannot be reported."""
    from mirt.estimation.base import _ITEM_PARAMETER_BOUNDS as bounds
    if model not in ("Rasch", "1PL", "2PL", "3PL", "4PL"):
        raise ValueError("Yalnız ikili modeller.")
    names = [str(i) for i in item_names]
    if not bool(getattr(fit, "converged", False)):
        return None, "Kestirim yakınsamadı; SH ve Wald GA üretilmedi."
    covariance = _matrix(fit.vcov)
    labels = list(fit.vcov_labels() if callable(fit.vcov_labels) else fit.vcov_labels)
    if covariance.ndim != 2 or covariance.shape != (len(labels), len(labels)) or not labels:
        return None, "Kütüphane kovaryans matrisi okunamadı; SH üretilmedi."
    if len(labels) > 2 ** min(len(names), 60) - 1:
        return None, "Serbest parametre sayısı gözlenebilir yanıt örüntüsü sayısını aşıyor; SH anlamsız."
    if not np.isfinite(covariance).all():
        return None, "Kovaryans matrisi sonlu değil (çoğunlukla sınırdaki parametreler); SH üretilmedi."
    covariance = (covariance + covariance.T) / 2
    eigen = np.linalg.eigvalsh(covariance)
    if eigen[0] <= 0:
        return None, "Kovaryans matrisi pozitif tanımlı değil; SH üretilemez."
    if eigen[-1] / eigen[0] > 1e8:
        return None, f"Kovaryans matrisi kötü koşullanmış (koşul sayısı {eigen[-1] / eigen[0]:.3g}); SH üretilmedi."
    coef = fit.coef()
    library_items = [str(i) for i in coef.index]
    if len(library_items) != len(names):
        return None, "Madde sayısı kütüphane katsayılarıyla uyuşmuyor."
    rows = []
    for position, label in enumerate(labels):
        parameter, _, item = label.partition("[")
        item = item.rstrip("]")
        if parameter not in NAMES or item not in library_items:
            return None, f"Tanınmayan kovaryans etiketi: {label}"
        j = library_items.index(item)
        estimate = float(coef.iloc[j][parameter])
        se = float(np.sqrt(covariance[position, position]))
        low, high = bounds[parameter]
        on_bound = min(abs(estimate - low), abs(estimate - high)) <= 1e-6
        note = "Kutu sınırında; Wald GA geçersiz" if on_bound else ""
        rows.append({"Madde": names[j], "Parametre": NAMES[parameter], "Kestirim": estimate,
                     "SH": np.nan if on_bound else se,
                     "GA %95 alt": np.nan if on_bound else estimate - Z95 * se,
                     "GA %95 üst": np.nan if on_bound else estimate + Z95 * se, "Not": note})
    order = {"a": 0, "b": 1, "c": 2, "d": 3}
    table = pd.DataFrame(rows)
    table["_o"] = table["Parametre"].map(order)
    table["_i"] = table["Madde"].map({n: i for i, n in enumerate(names)})
    return table.sort_values(["_i", "_o"]).drop(columns=["_o", "_i"]).reset_index(drop=True), None
