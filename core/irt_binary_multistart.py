"""Deterministic, bounded multi-start for Python mirt 3PL/4PL (library EM options only).

Candidates (fixed): the library default start, SQUAREM-accelerated EM, and six start points
c0 in {.05, .25, .40} x a0 in {.7, 1.5} (d0 = .9 for 4PL); all other settings as the screen fit.
Decision (fixed before use): among converged candidates the highest objective wins (log-likelihood for ML,
log-posterior for MAP); within 1e-6 the earlier candidate is kept, so the default wins ties. Reported per
candidate: convergence, objective gap to the best, maximum IRF difference to the best on [-4, 4].
Flags: the default is replaced when it is more than 1e-3 below the best; weak identification is flagged when
a converged candidate within 1e-2 of the best objective has IRF differing by more than .02;
converged candidates more than 1e-2 below the best are reported as separate (local) solutions.
"""
import numpy as np
import pandas as pd
from scipy.special import expit

from core.irt_python_mirt_adapter import fit_python_mirt

STARTS = tuple((c0, a0) for c0 in (.05, .25, .40) for a0 in (.7, 1.5))
REPLACE_GAP, NEAR_GAP, IRF_LIMIT = 1e-3, 1e-2, .02
THETA = np.linspace(-4, 4, 81)


def objective(fit, estimation):
    value = fit.log_posterior if estimation == "MAP" else fit.log_likelihood
    return float(value) if value is not None else float(fit.log_likelihood)


def _irf(fit, model):
    c = fit.coef()
    d = c["upper"].to_numpy(float) if model == "4PL" else 1.
    return c["guessing"].to_numpy(float) + (d - c["guessing"].to_numpy(float)) * expit(
        c["discrimination"].to_numpy(float) * (THETA[:, None] - c["difficulty"].to_numpy(float)))


def multistart(x, model, estimation="MAP", **solver):
    if model not in ("3PL", "4PL"):
        raise ValueError("Çoklu başlangıç yalnız 3PL/4PL içindir.")
    x = np.asarray(x); k = x.shape[1]
    specs = [("Varsayılan", {}), ("SQUAREM", {"accelerate": "squarem"})]
    for c0, a0 in STARTS:
        start = {"discrimination": np.full(k, a0), "guessing": np.full(k, c0)}
        if model == "4PL":
            start["upper"] = np.full(k, .9)
        specs.append((f"c0={c0:.2f}, a0={a0:.1f}", {"start_values": start}))
    fits = [(name, fit_python_mirt(x, model=model, estimation=estimation, **solver, **extra)) for name, extra in specs]
    ok = [i for i, (_, r) in enumerate(fits) if r.fit.converged]
    if not ok:
        return fits[0][1], pd.DataFrame(), ["Hiçbir başlangıç yakınsamadı; sonuç geçerli değildir."]
    values = [objective(r.fit, estimation) for _, r in fits]
    best = max(ok, key=lambda i: (values[i], -i))
    for i in ok:                                    # earliest candidate within 1e-6 of the best is kept
        if values[i] >= values[best] - 1e-6:
            best = i; break
    curve = _irf(fits[best][1].fit, model)
    rows = [{"Başlangıç": name, "Yakınsadı": bool(r.fit.converged), "Amaç": values[i],
             "En iyiden fark": values[best] - values[i] if r.fit.converged else np.nan,
             "En büyük IRF farkı": float(np.abs(_irf(r.fit, model) - curve).max()) if r.fit.converged else np.nan,
             "Seçildi": i == best} for i, (name, r) in enumerate(fits)]
    table = pd.DataFrame(rows)
    notes = []
    if not fits[0][1].fit.converged or values[0] < values[best] - REPLACE_GAP:
        notes.append(f"Varsayılan başlangıç en iyi çözümün {values[best] - values[0]:.4f} altında kaldı; "
                     f"'{fits[best][0]}' çözümü kullanıldı.")
    local = table[table["Yakınsadı"] & (table["En iyiden fark"] > NEAR_GAP)]
    if len(local):
        notes.append(f"{len(local)} başlangıç daha düşük amaç değerli ayrı çözümde durdu (en büyük fark "
                     f"{local['En iyiden fark'].max():.3f}); en yüksek amaç değerli çözüm seçildi.")
    near = table[table["Yakınsadı"] & (table["En iyiden fark"] <= NEAR_GAP)]
    if (near["En büyük IRF farkı"] > IRF_LIMIT).any():
        notes.append("Zayıf tanımlanma: neredeyse eşit amaç değerli çözümler farklı madde eğrileri veriyor "
                     f"(IRF farkı {near['En büyük IRF farkı'].max():.3f} > {IRF_LIMIT}).")
    return fits[best][1], table, notes
