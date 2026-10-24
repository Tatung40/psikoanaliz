"""B14 completion: person scores, information functions and model comparison for GRM/PCM/GPCM/NRM.

All quantities come from the fitted Python mirt model object (no reconstructed coefficients):
  * persons: mirt.fscores EAP with posterior SD (NRM: the latent axis sign is the library's; it is
    not a success direction);
  * information: model.information(theta, item_idx) per item, test information as the sum and
    SEM = 1/sqrt(test information); independently checked against sum_c (dP_c/dtheta)^2 / P_c
    (tests/test_irt_b14_persons_info.py and validation/evidence/p4_polytomous);
  * comparison: logLik/AIC/BIC/npar per ordinal model on the same coded data. Only PCM within GPCM
    is nested (same N(0,1) latent scale); its LR test is reported only when both fits converged.
    GRM vs PCM/GPCM is non-nested: information criteria only.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2


def person_scores(result, method="EAP"):
    import mirt
    if method not in ("EAP", "MAP"):
        raise ValueError("Yöntem EAP veya MAP olmalıdır.")
    x = np.asarray(result["coded"], int)
    sc = mirt.fscores(result["result"], x, method=method)
    theta = np.asarray(sc.theta, float).reshape(-1); se = np.asarray(sc.standard_error, float).reshape(-1)
    if len(theta) != len(x):
        raise ValueError("Kişi kestirimi sayısı veri satırlarıyla uyuşmuyor.")
    out = pd.DataFrame({"Sıra": np.arange(1, len(x) + 1), "θ": theta, "SH(θ)": se, "Yöntem": f"{method} (Python mirt fscores)"})
    if result["model"] != "NRM":
        out.insert(1, "Ham puan", x.sum(1))
    return out


def information_table(result, theta=None):
    model = result["result"].model
    theta = np.linspace(-4, 4, 161) if theta is None else np.asarray(theta, float)
    items = list(result["items"])
    info = np.column_stack([np.asarray(model.information(theta[:, None], item_idx=j), float).reshape(-1)
                            for j in range(len(items))])
    if not np.isfinite(info).all() or (info < -1e-10).any():
        raise ValueError("Geçersiz bilgi fonksiyonu değerleri.")
    frame = pd.DataFrame(info, columns=[f"I_{i}" for i in items]); frame.insert(0, "θ", theta)
    frame["Test bilgisi"] = info.sum(1)
    frame["SEM(θ)"] = 1 / np.sqrt(np.maximum(frame["Test bilgisi"], 1e-300))
    return frame


def independent_information(result, theta, h=1e-5):
    """Fisher information sum_c (dP_c/dtheta)^2 / P_c from model.probability by central differences."""
    model = result["result"].model
    P = lambda t: np.asarray(model.probability(np.asarray(t, float)[:, None]), float)
    dP = (P(theta + h) - P(theta - h)) / (2 * h)
    return (dP ** 2 / np.maximum(P(theta), 1e-300)).sum(2)


def model_comparison(coded, models=("GRM", "PCM", "GPCM")):
    from core.irt_python_mirt_adapter import fit_python_mirt
    x = np.asarray(coded, int); C = int(x.max()) + 1
    rows, fits = [], {}
    for m in models:
        try:
            f = fit_python_mirt(x, model=m, n_categories=C).fit
            fits[m] = f
            rows.append({"Model": m, "logLik": float(f.log_likelihood), "Parametre": int(f.n_parameters),
                         "AIC": float(f.aic), "BIC": float(f.bic), "Yakınsama": bool(f.converged), "Hata": ""})
        except Exception as exc:
            rows.append({"Model": m, "logLik": np.nan, "Parametre": np.nan, "AIC": np.nan, "BIC": np.nan,
                         "Yakınsama": False, "Hata": str(exc)[:200]})
    table = pd.DataFrame(rows)
    ok = table[table["Yakınsama"]]
    table["En küçük AIC"] = table["Model"] == (ok.loc[ok["AIC"].idxmin(), "Model"] if len(ok) else None)
    table["En küçük BIC"] = table["Model"] == (ok.loc[ok["BIC"].idxmin(), "Model"] if len(ok) else None)
    lr = None
    if "PCM" in fits and "GPCM" in fits and fits["PCM"].converged and fits["GPCM"].converged:
        stat = 2 * (fits["GPCM"].log_likelihood - fits["PCM"].log_likelihood)
        df = int(fits["GPCM"].n_parameters - fits["PCM"].n_parameters)
        if stat >= -1e-6 and df > 0:
            lr = {"Karşılaştırma": "PCM ⊂ GPCM (eğimler eşit = 1)", "LR": max(stat, 0.), "sd": df,
                  "p": float(chi2.sf(max(stat, 0.), df))}
    return table, lr
