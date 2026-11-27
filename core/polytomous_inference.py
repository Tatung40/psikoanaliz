"""Polytomous (GRM/PCM/GPCM/NRM) item-parameter SE and 95% Wald CI table with P4-C1 scope labels.
Same estimate/SE/CI construction as validation/polytomous_mc_study.py (est +/- 1.96 SE; PCM fixed a omitted; NRM
oriented as in the study, reference category omitted). Labels from core.polytomous_certification.ci_status."""
import numpy as np
import pandas as pd
from core.polytomous_certification import ci_status

MODELS = ("GRM", "PCM", "GPCM", "NRM")


def _orient_nrm(p):
    A = np.asarray(p["slopes"], float); D = np.asarray(p["intercepts"], float)
    A = A - A[:, :1]; D = D - D[:, :1]
    return ({"slopes": -A, "intercepts": D}, True) if A[0, -1] < 0 else ({"slopes": A, "intercepts": D}, False)


def parameter_table(outcome, item_names):
    """(DataFrame, reason). None for unsupported models or non-converged fits."""
    model = getattr(outcome, "model", None)
    if model not in MODELS:
        return None, "Politomöz parametre SH/GA yalnız GRM/PCM/GPCM/NRM için üretilir."
    fit = outcome.fit
    if not getattr(fit, "converged", False):
        return None, "Kestirim yakınsamadı; SH/GA üretilmedi."
    est = {k: np.asarray(v, float) for k, v in fit.model.parameters.items()}
    se = {k: np.asarray(v, float) for k, v in (fit.standard_errors or {}).items()}
    if model == "NRM":
        est, _ = _orient_nrm(est)
    rows = []
    for block, ev in est.items():
        if block == "discrimination" and model == "PCM":
            continue
        s = se.get(block, np.full(ev.shape, np.nan))
        status = ci_status(model, block, outcome.n_persons, getattr(outcome, "estimation", None), getattr(outcome, "solver_settings", None),
                           getattr(fit, "se_method", None), outcome.n_items, outcome.n_categories)
        e2, s2 = (ev, s) if ev.ndim == 2 else (ev[:, None], s[:, None])
        start = 1 if model == "NRM" else 0
        for j in range(e2.shape[0]):
            for c in range(start, e2.shape[1]):
                sv = float(s2[j, c]); ok = np.isfinite(sv) and sv > 0
                rows.append({"Madde": str(item_names[j]), "Parametre": block, "İndeks": c, "Kestirim": float(e2[j, c]),
                             "SH": sv if ok else np.nan, "%95 GA alt": float(e2[j, c] - 1.96 * sv) if ok else np.nan,
                             "%95 GA üst": float(e2[j, c] + 1.96 * sv) if ok else np.nan,
                             "SH yöntemi": getattr(fit, "se_method", None) or "kayıtsız", "Bilimsel durum (GA)": status if ok else "NOT CERTIFIED"})
    return pd.DataFrame(rows), None


def export_bundle(table, report=None):
    """(xlsx bytes, zip bytes) of the SAME table object shown on screen (plus model parameters if given)."""
    import io, zipfile
    sheets = {"Parametre_SH_GA": table}
    if report is not None and getattr(report, "parameters", None) is not None: sheets["Madde_Parametreleri"] = report.parameters
    xb = io.BytesIO()
    with pd.ExcelWriter(xb, engine="openpyxl") as w:
        for name, frame in sheets.items(): frame.to_excel(w, sheet_name=name, index=False)
    zb = io.BytesIO()
    with zipfile.ZipFile(zb, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("Parametre_SH_GA.csv", table.to_csv(index=False)); z.writestr("Parametre_SH_GA.xlsx", xb.getvalue())
        z.writestr("KAPSAM.txt", "SCOPED CERTIFIED only within P4-C1 (validation/p4_confirm/PROTOCOL.json): N=2000, 10 items, 4 categories, "
                                 "ML, 21 nodes, max_iter 500, tol 1e-4, NRM Oakes SE. Row labels say NOT CERTIFIED otherwise.\n")
    return xb.getvalue(), zb.getvalue()
