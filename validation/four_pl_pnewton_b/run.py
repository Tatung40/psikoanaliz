"""Execute PROTOCOL.json once. PYTHONPATH=. python -m validation.four_pl_pnewton_b.run --protocol-commit SHA"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np, pandas as pd

from validation.q1_q3.profile import ROOT, gh
from validation.r_mirt_phase1.equivalence_audit import BOUNDS, evaluate
from validation.r_mirt_phase1.information_audit import information
from validation.r_mirt_phase1.m12_diagnostic import feasible_scores, scores
from validation.four_pl_pnewton_b.algorithm import maximize

HERE = Path(__file__).resolve().parent; B = ROOT / "validation/four_pl_rescue"


def run(commit):
    assert hashlib.sha256((B / "design_b_data.csv").read_bytes()).hexdigest().startswith("0a928a9c")
    x = pd.read_csv(B / "design_b_data.csv"); X = x.to_numpy(float); cols = list(BOUNDS)
    p0 = pd.DataFrame(json.loads((B / "verdict_B.json").read_text())["parameters"]).T.loc[x.columns, cols].astype(float)
    J = len(p0); lo = np.tile([b[0] for b in BOUNDS.values()], J); hi = np.tile([b[1] for b in BOUNDS.values()], J)
    frame = lambda v: pd.DataFrame(v.reshape(J, 4), index=p0.index, columns=cols)
    z, w = gh(161); az, aw = gh(241); step = float(np.cbrt(np.finfo(float).eps))
    f = lambda v: evaluate(X, frame(v), z, w)[2]
    g = lambda v: scores(X, frame(v), z, w).to_numpy().ravel()
    H = lambda v: information(lambda q: scores(X, q, z, w), frame(v), step)[0]
    log = []; t0 = time.monotonic()
    v, fv, hist = maximize(f, g, H, p0.to_numpy().ravel(), lo, hi, log=lambda r: print({k: (round(val, 9) if isinstance(val, float) else val) for k, val in r.items()}, flush=True))
    seconds = time.monotonic() - t0; p = frame(v)
    proj = lambda q, gg: float(np.abs(feasible_scores(q, gg).to_numpy()).max())
    g161, g241 = scores(X, p, z, w), scores(X, p, az, aw); ll161, ll241 = f(v), evaluate(X, p, az, aw)[2]
    bnd = [dict(item=i, parameter=k, value=float(p.loc[i, k]), side="lower" if p.loc[i, k] == BOUNDS[k][0] else "upper",
                score=float(g161.loc[i, k])) for i in p.index for k in cols if p.loc[i, k] in BOUNDS[k]]
    for b in bnd: b["kkt_sign_ok"] = (b["score"] <= 0) if b["side"] == "lower" else (b["score"] >= 0)
    keys = {(b["item"], b["parameter"]) for b in bnd}
    h = H(v); h = (h + h.T) / 2; free = [j for j, (i, k) in enumerate((i, k) for i in p.index for k in cols) if (i, k) not in keys]
    ev = np.linalg.eigvalsh(h[np.ix_(free, free)]); cond = float(ev[-1] / ev[0]) if ev[0] > 0 else float("inf")
    s161, s241 = proj(p, g161), proj(p, g241)
    gates = dict(GH161=s161 <= 1e-5, GH241=s241 <= 1e-5, audit_loglik=abs(ll241 - ll161) <= 1e-3, information=bool(ev[0] > 0 and cond <= 1e8))
    numeric = all(gates.values()); regular = numeric and not bnd
    out = dict(protocol_commit=commit, status="DIAGNOSTIC / NOT CERTIFIED", monte_carlo_executions=0, seconds=seconds,
               start=dict(score=hist[0]["score"], logLik=hist[0]["f"]), history=hist,
               accepted_steps=sum("t" in r for r in hist), stop=hist[-1].get("stop"),
               final=dict(score_GH161=s161, score_GH241=s241, logLik_GH161=ll161, logLik_GH241=ll241,
                          min_eigenvalue=float(ev[0]), condition=cond), boundaries=bnd, gates=gates,
               classification="NUMERICAL PASS" if regular else ("BOUNDARY SOLUTION" if numeric else "NUMERICAL FAIL"),
               numerical_pass=bool(regular), max_abs_change_from_R=float((p - p0).abs().to_numpy().max()), parameters=p.to_dict("index"))
    if regular:
        t = pd.read_csv(B / "design_b_truth.csv").set_index("item").loc[p.index, cols]
        se = frame(np.sqrt(np.diag(np.linalg.inv(h)))); R = json.loads((B / "PROTOCOL.json").read_text())["recovery"]
        rmse = {k: float(np.sqrt(((p[k] - t[k]) ** 2).mean())) for k in cols}; mz = float(((p - t) / se).abs().to_numpy().max())
        out["recovery"] = dict(rmse=rmse, max_abs_z=mz, passed=bool(all(rmse[k] <= R["rmse_max"][k] for k in cols) and mz <= R["max_abs_z"]))
    (HERE / "result.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--protocol-commit", required=True); r = run(ap.parse_args().protocol_commit)
    print(json.dumps({k: r[k] for k in ("seconds", "start", "accepted_steps", "stop", "final", "gates", "classification", "max_abs_change_from_R") }, indent=1))
    print("boundaries", r["boundaries"]); print("recovery", r.get("recovery"))
