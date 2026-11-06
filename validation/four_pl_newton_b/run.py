"""Execute PROTOCOL.json once. PYTHONPATH=. python -m validation.four_pl_newton_b.run --protocol-commit SHA"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np, pandas as pd

from validation.q1_q3.profile import ROOT, gh
from validation.r_mirt_phase1.equivalence_audit import BOUNDS, evaluate
from validation.r_mirt_phase1.information_audit import information
from validation.r_mirt_phase1.m12_diagnostic import feasible_scores, scores

HERE = Path(__file__).resolve().parent; P = json.loads((HERE / "PROTOCOL.json").read_text())
LO = [b[0] for b in BOUNDS.values()]; HI = [b[1] for b in BOUNDS.values()]


def run(commit):
    for k in ("path", "truth"):
        assert hashlib.sha256((ROOT / P["data"][k]).read_bytes()).hexdigest() == P["data"]["sha256" if k == "path" else "truth_sha256"]
    x = pd.read_csv(ROOT / P["data"]["path"]); X = x.to_numpy(float)
    vb = json.loads((ROOT / "validation/four_pl_rescue/verdict_B.json").read_text())
    p0 = pd.DataFrame(vb["parameters"]).T.loc[x.columns, list(BOUNDS)].astype(float)
    z, w = gh(161); az, aw = gh(241)
    value = lambda v: evaluate(X, v, z, w)[2]; score = lambda v: scores(X, v, z, w)
    projected = lambda v, g=None: float(np.abs(feasible_scores(v, score(v) if g is None else g).to_numpy()).max())
    r_stage = vb["stages"][-1]; mapping = abs(r_stage["r_loglik"] - value(p0))
    checks, g0 = [], score(p0)                            # analytic vs central-difference gradient
    for item in ("I01", "I10", "I20"):
        for k in BOUNDS:
            h = 1e-5; up, dn = p0.copy(), p0.copy(); up.loc[item, k] += h; dn.loc[item, k] -= h
            num = (value(up) - value(dn)) / (2 * h); ana = float(g0.loc[item, k])
            checks.append(dict(item=item, parameter=k, analytic=ana, numeric=num, rel=abs(ana - num) / max(1., abs(ana))))
    pre = dict(mapping_abs=mapping, mapping_ok=mapping <= 1e-6, gradient_max_rel=max(c["rel"] for c in checks), gradient_checks=checks)
    pre["ok"] = bool(pre["mapping_ok"] and pre["gradient_max_rel"] <= 1e-4)
    out = dict(protocol_commit=commit, status="DIAGNOSTIC / NOT CERTIFIED", monte_carlo_executions=0, prechecks=pre)
    p, history, started = p0.copy(), [], time.monotonic()
    if pre["ok"]:
        N = P["newton"]
        for step in range(N["max_steps"] + 1):
            g = score(p).to_numpy().ravel(); v = p.to_numpy().ravel()
            rec = dict(step=step, elapsed=time.monotonic() - started, score=projected(p, score(p)), logLik=value(p)); history.append(rec)
            if rec["score"] <= P["gates"]["GH161_projected_score"]: rec["stop"] = "STATIONARY"; break
            if rec["elapsed"] > 15 * 60: rec["stop"] = "BUDGET"; break
            if step == N["max_steps"]: rec["stop"] = "MAX STEPS"; break
            free = [j for j, q in enumerate(v) if not ((q == LO[j % 4] and g[j] <= 0) or (q == HI[j % 4] and g[j] >= 0))]
            h, _ = information(score, p, float(np.cbrt(np.finfo(float).eps))); h = (h + h.T) / 2
            face = h[np.ix_(free, free)]; eig = np.linalg.eigvalsh(face); rec.update(information_min=float(eig[0]), condition=float(np.linalg.cond(face)))
            if eig[0] <= 0 or rec["condition"] > 1e8: rec["stop"] = "INFORMATION GATE"; break
            d = np.zeros(len(v)); d[free] = np.linalg.solve(face, g[free]); ok = False
            for k in range(10):
                nv = v + d * .5 ** k
                if any(nv[j] < LO[j % 4] or nv[j] > HI[j % 4] for j in range(len(nv))): continue
                cand = pd.DataFrame(nv.reshape(p.shape), index=p.index, columns=p.columns)
                if value(cand) >= rec["logLik"] and projected(cand) < rec["score"]:
                    p, ok = cand, True; rec["accepted_backtrack"] = k; break
            if not ok: rec["stop"] = "STAGNATION"; break
    seconds = time.monotonic() - started
    ll161, ll241 = value(p), evaluate(X, p, az, aw)[2]; s161, s241 = projected(p), projected(p, scores(X, p, az, aw))
    bnd = [f"{i}:{k}" for i in p.index for k in BOUNDS if p.loc[i, k] in BOUNDS[k]]
    h, _ = information(score, p, float(np.cbrt(np.finfo(float).eps))); h = (h + h.T) / 2
    free = [j for j, (i, k) in enumerate((i, k) for i in p.index for k in BOUNDS) if f"{i}:{k}" not in bnd]
    eig = np.linalg.eigvalsh(h[np.ix_(free, free)]); cond = float(eig[-1] / eig[0]) if eig[0] > 0 else float("inf")
    gates = dict(GH161=s161 <= 1e-5, GH241=s241 <= 1e-5, audit_loglik=abs(ll241 - ll161) <= 1e-3,
                 information=bool(eig[0] > 0 and cond <= 1e8), zero_boundaries=not bnd)
    out.update(seconds=seconds, history=history, start=dict(score=projected(p0), logLik=value(p0)),
               final=dict(score_GH161=s161, score_GH241=s241, logLik_GH161=ll161, logLik_GH241=ll241,
                          min_eigenvalue=float(eig[0]), condition=cond, boundaries=bnd),
               gates=gates, numerical_pass=bool(pre["ok"] and all(gates.values())), parameters=p.to_dict("index"),
               max_abs_change_from_R=float((p - p0).abs().to_numpy().max()))
    if out["numerical_pass"]:
        t = pd.read_csv(ROOT / P["data"]["truth"]).set_index("item").loc[p.index, list(BOUNDS)]
        se = pd.DataFrame(np.sqrt(np.diag(np.linalg.inv(h))).reshape(p.shape), index=p.index, columns=p.columns)
        R = json.loads((ROOT / "validation/four_pl_rescue/PROTOCOL.json").read_text())["recovery"]
        rmse = {k: float(np.sqrt(((p[k] - t[k]) ** 2).mean())) for k in BOUNDS}; mz = float(((p - t) / se).abs().to_numpy().max())
        out["recovery"] = dict(rmse=rmse, max_abs_z=mz, mean_se=se.mean().to_dict(),
                               passed=bool(all(rmse[k] <= R["rmse_max"][k] for k in BOUNDS) and mz <= R["max_abs_z"]))
    (HERE / "result.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--protocol-commit", required=True); r = run(ap.parse_args().protocol_commit)
    print(json.dumps({k: r[k] for k in ("seconds", "start", "final", "gates", "numerical_pass", "max_abs_change_from_R") if k in r}, indent=1))
    print("pre", r["prechecks"]["mapping_abs"], r["prechecks"]["gradient_max_rel"], r.get("recovery"))
    print([(h["step"], "%.3g" % h["score"], round(h["logLik"], 6), h.get("accepted_backtrack"), h.get("stop")) for h in r["history"]])
