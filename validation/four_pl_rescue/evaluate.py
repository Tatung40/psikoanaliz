"""Common-evaluator verdict for the R mirt rescue fits (PROTOCOL.json). No refits.
PYTHONPATH=. python -m validation.four_pl_rescue.evaluate --design A|B --fits DIR --protocol-commit SHA
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

from validation.q1_q3.profile import ROOT, gh
from validation.r_mirt_phase1.equivalence_audit import BOUNDS, evaluate
from validation.r_mirt_phase1.information_audit import information
from validation.r_mirt_phase1.m12_diagnostic import feasible_scores, scores

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "PROTOCOL.json").read_text())
G = P["numerical_gates"]


def load(design):
    d = P["designs"][design]; x = pd.read_csv(ROOT / d["data"]); return x, x.to_numpy(float)


def params(path, columns):
    p = pd.read_csv(path).set_index("item").loc[columns, list(BOUNDS)]
    for k, (lo, hi) in BOUNDS.items():                    # preregistered snap of asymptotic bounds (L-BFGS-B on logit scale)
        v = p[k].to_numpy(); v[np.abs(v - lo) <= G["snap"]] = lo; v[np.abs(v - hi) <= G["snap"]] = hi; p[k] = v
    return p


def projected(X, p, n):
    z, w = gh(n); return float(np.abs(feasible_scores(p, scores(X, p, z, w)).to_numpy()).max())


def verdict(design, fits, commit):
    x, X = load(design); meta = pd.read_csv(Path(fits) / "metadata.csv")
    stages = [dict(stage=r.stage, quadpts=int(r.quadpts), r_converged=bool(r.converged), iterations=int(r.iterations),
                   seconds=float(r.seconds), r_loglik=float(r.loglik)) for r in meta.itertuples()]
    for s in stages:
        p = params(Path(fits) / f"{s['stage']}.csv", x.columns)
        s.update(common_loglik=evaluate(X, p, *gh(s["quadpts"]))[2], projected_score=projected(X, p, s["quadpts"]))
    p = params(Path(fits) / f"{P['objective_stage']}.csv", x.columns); q, qa = P["objective_grid"], P["audit_grid"]
    inside = bool(all(((p[k] >= lo) & (p[k] <= hi)).all() for k, (lo, hi) in BOUNDS.items()))
    ll, lla = evaluate(X, p, *gh(q))[2], evaluate(X, p, *gh(qa))[2]
    s_obj, s_aud = projected(X, p, q), projected(X, p, qa)
    bnd = [f"{i}:{k}" for i in p.index for k, (lo, hi) in BOUNDS.items() if p.loc[i, k] in (lo, hi)]
    z, w = gh(q); h, _ = information(lambda v: scores(X, v, z, w), p, float(np.cbrt(np.finfo(float).eps))); h = (h + h.T) / 2
    free = [j for j, (i, k) in enumerate((i, k) for i in p.index for k in BOUNDS) if f"{i}:{k}" not in bnd]
    face = h[np.ix_(free, free)]; eig = np.linalg.eigvalsh(face); cond = float(eig[-1] / eig[0]) if eig[0] > 0 else float("inf")
    gates = dict(inside_bounds=inside, stationary_objective=s_obj <= G["projected_score"],
                 audit=bool(s_aud <= G["projected_score"] and abs(lla - ll) <= G["audit_loglik"]),
                 information=bool(eig[0] > 0 and cond <= G["condition"]))
    if not inside: cls = "OUTSIDE COMMON BOUNDS"
    elif not gates["stationary_objective"]: cls = "CONVERGENCE FAILURE"
    elif not gates["audit"]: cls = "QUADRATURE ERROR"
    elif not gates["information"]: cls = "WEAK IDENTIFICATION (information gate)"
    elif bnd: cls = "BOUNDARY SOLUTION (numerically stationary; non-regular, no Wald inference)"
    else: cls = "NUMERICAL PASS"
    out = dict(design=design, protocol_commit=commit, stages=stages, objective=dict(grid=q, loglik=ll, projected_score=s_obj),
               audit=dict(grid=qa, loglik=lla, projected_score=s_aud, delta_loglik=lla - ll),
               information=dict(min_eigenvalue=float(eig[0]), condition=cond), boundaries=bnd, gates=gates,
               numerical_pass=bool(all(gates.values()) and not bnd), classification=cls, parameters=p.to_dict("index"))
    if design == "A":
        py = json.loads((ROOT / "validation/four_pl_gh_refinement/result.json").read_text())
        pp = pd.DataFrame(py["parameters"]).T.loc[p.index, list(BOUNDS)]
        out["vs_python_pr145"] = dict(python_loglik_GH161=py["logLik"]["GH161"], r_minus_python_GH161=evaluate(X, p, *gh(161))[2] - py["logLik"]["GH161"],
                                      max_abs_param=float((p - pp).abs().to_numpy().max()), python_boundaries=py["boundaries"])
    if design == "B" and out["numerical_pass"]:
        t = pd.read_csv(ROOT / P["designs"]["B"]["truth"]).set_index("item").loc[p.index, list(BOUNDS)]
        se = pd.DataFrame(np.sqrt(np.diag(np.linalg.inv(h))).reshape(p.shape), index=p.index, columns=p.columns)
        zt = (p - t) / se; R = P["recovery"]
        rmse = {k: float(np.sqrt(((p[k] - t[k]) ** 2).mean())) for k in BOUNDS}
        out["recovery"] = dict(rmse=rmse, max_abs_z=float(zt.abs().to_numpy().max()), mean_se=se.mean().to_dict(),
                               pass_=bool(all(rmse[k] <= R["rmse_max"][k] for k in BOUNDS) and zt.abs().to_numpy().max() <= R["max_abs_z"]))
    (Path(fits) / f"verdict_{design}.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--design"); ap.add_argument("--fits"); ap.add_argument("--protocol-commit")
    a = ap.parse_args(); r = verdict(a.design, a.fits, a.protocol_commit)
    print(json.dumps({k: v for k, v in r.items() if k != "parameters"}, indent=1))
