"""Advanced MIRT reports on the shared {metadata, tables} contract (Codex report/export infrastructure reused).

Nothing here is certified. The routes are EXPERIMENTAL / NOT CERTIFIED, and item SE/CI/p, fit inference
and Wilks comparisons are not produced.
"""
import numpy as np
import pandas as pd

from core import irt_multidim_general as engine
from core import irt_multidim_general_report as general
from core import mirt_advanced as adv
from core import mirt_r_backend as rb
from core.irt_multidim import STATUS, report, surface_tables

BIFACTOR_MODELS = {"GRM bifactor — ortogonal G+S (Python, deneysel)": "GRM",
                   "GPCM bifactor — ortogonal G+S (Python, deneysel)": "GPCM"}
ENGINES = ("SciPy (varsayılan)", "Native R mirt 1.48 (isteğe bağlı)")
WITHHELD = ("Item SE/CI", "Global/item-fit p", "Wilks comparison")
BIFACTOR_LIMITATION = (
    "EXPERIMENTAL / NOT CERTIFIED. Polytomous bifactor model: a confirmatory orthogonal MGRM/MGPCM (G + exactly 3 specific factors, "
    ">=3 items each, positive loadings, unit variances) on a full GH grid. Python mirt 1.2.0 has no polytomous bfactor API. "
    "Independent R agreement exists for a single recorded 4D fixture only (GPCM default; GRM only with the high-precision profile). "
    "Binary bifactor P8 recovery NOT MET evidence is not transferred. PCM bifactor WITHHELD. Item SE/CI/p and fit inference are disabled.")
HP_LIMITATION = (
    "EXPERIMENTAL / NOT CERTIFIED. Explicit GRM high-precision profile (tol 1e-11, max_iter 3000, item_optim_ftol 1e-14). "
    "Independent R probability agreement exists for the recorded GRM EFA 2D and GRM bifactor 4D fixtures only. No automatic trigger is "
    "used; logLik change is not a reliable stationarity diagnostic. Item SE/CI/p are disabled.")
R_LIMITATION = (
    "EXPERIMENTAL / NOT CERTIFIED. Optional native R mirt 1.48 engine for the fixed-asymptote general CFA 3PL/4PL. It reproduces archived "
    "direct R fits (adapter fidelity, not independent validation) and agrees with the independent SciPy engine on 6 recorded fixtures. "
    "Native R mirt is not available on Streamlit Community Cloud; the route refuses when it is missing and never falls back to another engine. "
    "Item SE/CI/p, LR and MAP are disabled.")


def _scope(meta, tables, limitation, numerical):
    meta.update(Status=STATUS, Numerical_status=numerical, Inferential_status="WITHHELD", Production_status="EXPERIMENTAL / NOT CERTIFIED",
                Limitations=limitation)
    rows = [dict(Output="Parameters / probabilities / information / EAP", Numerical_status=numerical, Inferential_status="WITHHELD",
                 Conditions="Recorded fixtures only", Qualification="Numerical agreement is not inferential certification")]
    rows += [dict(Output=n, Numerical_status="WITHHELD", Inferential_status="WITHHELD", Conditions="No validated method",
                  Qualification="Not supplied") for n in WITHHELD]
    tables["Output_scope"] = pd.DataFrame(rows)
    tables["Summary"] = pd.DataFrame([dict(Setting=str(k), Value=str(v)) for k, v in meta.items()])
    return dict(metadata=meta, tables=tables)


def bifactor_report(data, groups, label, *, high_precision=False, n_quadpts=7, max_iter=1000, tol=1e-7, person_ids=None):
    model = BIFACTOR_MODELS[label]
    outcome = adv.fit_polytomous_bifactor(data, np.asarray(groups, int), model, n_quadpts=n_quadpts, max_iter=max_iter, tol=tol,
                                          high_precision=high_precision)
    result = report(outcome, person_ids)
    meta, tables = dict(result["metadata"]), dict(result["tables"])
    meta.update(Model=label, Advanced_route="core.mirt_advanced.fit_polytomous_bifactor", Bifactor_groups=list(map(int, groups)),
                High_precision=bool(high_precision), Calibration_default_seconds=outcome.get("default_seconds"),
                Continuation_seconds=outcome.get("continuation_seconds"))
    if model == "GRM" and not high_precision:
        numerical = ("NOT VALIDATED: the default GRM bifactor profile FAILED the independent R probability gate; "
                     "R agreement exists only with the high-precision profile, which is disabled for this fit")
    elif model == "GRM":
        numerical = "EXPERIMENTAL (single-fixture R agreement with the high-precision profile)"
    else:
        numerical = "EXPERIMENTAL (single-fixture R agreement)"
    return _scope(meta, tables, BIFACTOR_LIMITATION, numerical)


def grm_high_precision_report(data, pattern, *, exploratory=False, n_quadpts=15, person_ids=None):
    outcome = adv.fit_grm_high_precision(data, np.asarray(pattern, int), exploratory=exploratory, n_quadpts=n_quadpts)
    result = report(outcome, person_ids)
    meta, tables = dict(result["metadata"]), dict(result["tables"])
    meta.update(Model="GRM (yüksek hassasiyet profili)", Precision_profile=outcome["profile"], **{f"HP_{k}": v for k, v in outcome["diagnostics"].items()})
    tables["Precision_diagnostics"] = pd.DataFrame([outcome["diagnostics"]])
    return _scope(meta, tables, HP_LIMITATION, "EXPERIMENTAL (R agreement on recorded GRM EFA 2D / bifactor 4D only)")


def native_r_report(data, model, pattern, *, c=.2, u=1., n_quadpts=15, max_iter=3000, tol=1e-8, person_ids=None, timeout=900):
    family, free_c, free_u = general.CONFIG[model]
    if free_c or free_u:
        raise ValueError("Native R engine: only fixed-asymptote profiles (free/mixed c/d are restricted)")
    x, spec = general.specification(data, model, pattern, "full", False, c, u)
    ids = list(x.index) if person_ids is None else list(person_ids)
    if len(ids) != len(x) or pd.Series(ids).isna().any() or not pd.Index(ids).is_unique:
        raise ValueError("Kimlikler benzersiz, eksiksiz ve satırlarla eşleşmeli")
    _, _, lower, upper, _ = spec.unpack(spec.start_bounds()[0])
    r = rb.fit(x, f"M{family}_fixed", pattern=spec.pattern.astype(int), c=lower, u=upper if family == "4PL" else None,
               n_quadpts=n_quadpts, max_iter=max_iter, tol=tol, timeout=timeout)
    v = r["values"]; items = list(x.columns)
    get = lambda it, name: float(v[(v.item == it) & (v.name == name)].value.iloc[0])
    A = np.array([[get(it, f"a{d+1}") for d in range(spec.q)] for it in items])
    values = np.concatenate([A[spec.pattern], [get(it, "d") for it in items], v[v.name.str.startswith("eta")].value.to_numpy(float)])
    if len(values) != spec.n_parameters or int(r["n_parameters"]) != spec.n_parameters:
        raise ValueError("R parameter count does not match the specification")
    objective = engine.Objective(x, spec, n_quadpts)
    loglik = -objective.evaluate(values)[0]
    A, delta, lo, up, eta = spec.unpack(values); C = spec.covariance(eta)
    meta = dict(Model=model, Backend="NATIVE_R_MIRT", Engine=f"R mirt {r['mirt_version']} ({r['R']}); optional native engine",
                Validation_scope=r["validation_scope"], N=len(x), Items=spec.j, Dimensions=spec.q, Labels=("0", "1"),
                Converged=bool(r["converged"]), Iterations=int(r["iterations"]), LogLik=float(r["logLik"]), Common_grid_logLik=loglik,
                AIC=float(r["AIC"]), BIC=float(r["BIC"]), Parameters=spec.n_parameters, Fixed_c=True, Fixed_d=True, n_quadpts=n_quadpts,
                max_iter=max_iter, tol=tol, R_seconds=float(r["seconds"]), Wall_seconds=float(r["wall_seconds"]), At_box_bound=False,
                Hessian="Not computed; SE/CI/LR disabled", MAP="NOT VALIDATED / DISABLED")
    cols = [f"F{d+1}" for d in range(spec.q)]
    tables = dict(Parameters=pd.DataFrame([dict(Item=it, delta=delta[i], c=lo[i], d=up[i], **{f"a{d+1}": A[i, d] for d in range(spec.q)}) for i, it in enumerate(items)]),
                  Fit=pd.DataFrame([dict(Converged=meta["Converged"], Iterations=meta["Iterations"], LogLik=meta["LogLik"], Common_grid_logLik=loglik,
                                         AIC=meta["AIC"], BIC=meta["BIC"], Free_parameters=spec.n_parameters)]),
                  Covariance=pd.DataFrame(C, index=cols, columns=cols).reset_index(names="Factor"),
                  Pattern=pd.DataFrame(spec.pattern.astype(int), index=x.columns, columns=cols).reset_index(names="Item"),
                  R_values=v)
    if meta["Converged"]:
        mean, cov = engine.eap(objective, values)
        scores = pd.DataFrame(mean, columns=cols); scores.insert(0, "ID", ids)
        for d in range(spec.q):
            scores[f"Posterior_SD_F{d+1}"] = np.sqrt(cov[:, d, d])
        tables["EAP"] = scores
        tables.update(surface_tables(general.Curves(spec, values), items, meta["Labels"], spec.q))
    return _scope(meta, tables, R_LIMITATION, "EXPERIMENTAL (adapter fidelity + SciPy agreement on 6 recorded fixtures)")
