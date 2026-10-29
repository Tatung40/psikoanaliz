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
STATUS = ("Oakes gözlenen bilgi SH ve %95 Wald GA. Önceden kayıtlı Monte Carlo kapsaması: 1PL/2PL (N = 1000, 12 madde) "
          ".945–.953; 3PL MAP (N = 3000, 30 madde) .950–.968. Ölçütü (.95 ± .02, sınırda tutulan oran ≤ .05) "
          "sağlamayan 3PL/4PL parametrelerinde SH gösterilir, Wald GA gösterilmez "
          "(validation/evidence/p6_binary_se, p7_binary34_se).")
# Pre-registered 3PL/4PL decision (validation/binary34_se_coverage_study.py): (model, estimation, parameter) ->
# (coverage farthest from .95 over the design cells, largest on-bound withheld rate, Wald CI supported).
EVIDENCE_34 = {
    ("3PL", "MAP", "a"): (0.950, 0.000, True),
    ("3PL", "MAP", "b"): (0.967, 0.000, True),
    ("3PL", "MAP", "c"): (0.968, 0.000, True),
    ("4PL", "MAP", "a"): (0.994, 0.000, False),
    ("4PL", "MAP", "b"): (0.971, 0.000, False),
    ("4PL", "MAP", "c"): (0.981, 0.000, False),
    ("4PL", "MAP", "d"): (0.979, 0.000, False),
    ("3PL", "ML", "a"): (0.929, 0.000, False),
    ("3PL", "ML", "b"): (0.827, 0.000, False),
    ("3PL", "ML", "c"): (0.936, 0.128, False),
    ("4PL", "ML", "a"): (0.936, 0.001, True),
    ("4PL", "ML", "b"): (0.815, 0.000, False),
    ("4PL", "ML", "c"): (0.884, 0.175, False),
    ("4PL", "ML", "d"): (0.869, 0.281, False)
}
# Same study: (largest |mean bias|, largest RMSE) over the cells. Coverage is not estimation accuracy: where
# |bias| exceeds the .05 limit used for 1PL/2PL (p6_binary_se), a kept CI carries an explicit warning.
ACCURACY_34 = {
    ("3PL", "MAP", "a"): (-0.022, 0.132),
    ("3PL", "MAP", "b"): (+0.004, 0.133),
    ("3PL", "MAP", "c"): (+0.000, 0.047),
    ("4PL", "MAP", "a"): (+0.036, 0.176),
    ("4PL", "MAP", "b"): (-0.019, 0.172),
    ("4PL", "MAP", "c"): (+0.005, 0.045),
    ("4PL", "MAP", "d"): (-0.018, 0.036),
    ("3PL", "ML", "a"): (+0.018, 0.159),
    ("3PL", "ML", "b"): (+0.008, 0.230),
    ("3PL", "ML", "c"): (-0.002, 0.082),
    ("4PL", "ML", "a"): (+0.179, 0.504),
    ("4PL", "ML", "b"): (+0.013, 0.293),
    ("4PL", "ML", "c"): (+0.010, 0.102),
    ("4PL", "ML", "d"): (-0.011, 0.068)
}
BIAS_LIMIT = 0.05


def _matrix(value):
    return np.asarray(value() if callable(value) else value, float)


def parameter_inference(fit, model, item_names, estimation=None):
    """Return (table, reason). table is None and reason explains why when SE cannot be reported.

    With estimation ("ML"/"MAP") a 3PL/4PL Wald CI is kept only where EVIDENCE_34 supports it; estimation=None
    returns the raw intervals (used by the coverage study itself)."""
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
    coef = fit.coef()
    library_items = [str(i) for i in coef.index]
    if len(library_items) != len(names):
        return None, "Madde sayısı kütüphane katsayılarıyla uyuşmuyor."
    parsed = []
    for label in labels:
        parameter, _, item = label.partition("[")
        item = item.rstrip("]")
        if parameter not in NAMES or item not in library_items:
            return None, f"Tanınmayan kovaryans etiketi: {label}"
        j = library_items.index(item)
        estimate = float(coef.iloc[j][parameter])
        low, high = bounds[parameter]
        parsed.append((j, parameter, estimate, min(abs(estimate - low), abs(estimate - high)) <= 1e-6))
    # The library holds coordinates on an optimizer bound fixed and returns NaN rows/columns for them; the
    # remaining block is the observed-information covariance of the interior parameters given those values.
    # A non-finite entry among interior parameters is a failure of the whole matrix.
    held = np.array([p[3] for p in parsed])
    active = ~held
    if not active.any():
        return None, "Tüm parametreler kutu sınırında; SH üretilemez."
    block = covariance[np.ix_(active, active)]
    if not np.isfinite(block).all():
        return None, "Kovaryans matrisi sınır dışı parametrelerde sonlu değil; SH üretilmedi."
    block = (block + block.T) / 2
    eigen = np.linalg.eigvalsh(block)
    if eigen[0] <= 0:
        return None, "Kovaryans matrisi pozitif tanımlı değil; SH üretilemez."
    if eigen[-1] / eigen[0] > 1e8:
        return None, f"Kovaryans matrisi kötü koşullanmış (koşul sayısı {eigen[-1] / eigen[0]:.3g}); SH üretilmedi."
    rows = []
    for position, (j, parameter, estimate, on_bound) in enumerate(parsed):
        se = np.nan if on_bound else float(np.sqrt(covariance[position, position]))
        rows.append({"Madde": names[j], "Parametre": NAMES[parameter], "Kestirim": estimate, "SH": se,
                     "GA %95 alt": estimate - Z95 * se, "GA %95 üst": estimate + Z95 * se,
                     "Not": "Kutu sınırında; SH/Wald GA geçersiz (kütüphane bu parametreyi sabit tutar)"
                     if on_bound else ""})
    if estimation is not None and model in ("3PL", "4PL"):
        for row in rows:
            coverage, withheld, supported = EVIDENCE_34[(model, estimation, row["Parametre"])]
            bias, rmse = ACCURACY_34[(model, estimation, row["Parametre"])]
            if row["Not"]:
                continue
            if not supported:
                row["GA %95 alt"] = row["GA %95 üst"] = np.nan
                row["Not"] = (f"Wald GA gösterilmedi: önceden kayıtlı MC kapsaması {coverage:.3f}, sınırda tutulan "
                              f"oran {withheld:.3f} (ölçüt .95 ± .02, ≤ .05)")
            elif abs(bias) > BIAS_LIMIT:
                row["Not"] = (f"Uyarı: GA kapsaması ölçütü sağladı ({coverage:.3f}) ancak MC'de sapma {bias:+.3f}, "
                              f"RMSE {rmse:.3f}; nokta kestirimi güvenilir değildir")
    order = {"a": 0, "b": 1, "c": 2, "d": 3}
    table = pd.DataFrame(rows)
    table["_o"] = table["Parametre"].map(order)
    table["_i"] = table["Madde"].map({n: i for i, n in enumerate(names)})
    return table.sort_values(["_i", "_o"]).drop(columns=["_o", "_i"]).reset_index(drop=True), None
