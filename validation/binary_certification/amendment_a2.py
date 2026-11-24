"""Amendment A2 of PROTOCOL.json: G2/G3 of all four models under the corrected binary default (61 GH nodes).

    PYTHONPATH=. python validation/binary_certification/amendment_a2.py --workers 2 --output validation/evidence/binary_certification/amendment_a2.json
"""
import argparse
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROTOCOL = json.loads((Path(__file__).with_name("PROTOCOL.json")).read_text())
SEED = PROTOCOL["amendment_A2"]["retest"]["seed"]
CORNERS = [(n, k) for n in (1000, 10000) for k in (12, 30)]
CORNERS34 = [(n, k) for n in (3000, 10000) for k in (12, 30)]
JOBS = ([("1PL fixed_a", n, k, 300 if n == 1000 else 100) for n, k in CORNERS]
        + [("1PL common_a", n, k, 1000) for n, k in CORNERS]
        + [("2PL", n, k, 300 if n == 1000 else 100) for n, k in CORNERS]
        + [("3PL MAP", n, k, 300 if n == 3000 else 150) for n, k in CORNERS34]
        + [("4PL MAP", n, k, 100 if n == 3000 else 50) for n, k in CORNERS34])
P1 = {"rmse_a_b": 0.30, "rmse_c_d": 0.07, "irf_rmse": 0.02, "at_bound_rate": 0.05}       # p1 pre-registered criteria


def _cell(job):
    from core.irt_binary_inference import parameter_inference
    from core.irt_python_mirt_adapter import fit_python_mirt
    from validation.binary_certification.common import simulate, truth
    from validation.coverage_precision import coverage_precision
    label, n, k, reps = job
    model, variant = (label.split() + [None])[:2]
    estimation = "MAP" if variant == "MAP" else "ML"; variant = None if variant == "MAP" else variant
    p = truth(model, k, variant); names = [f"M{j}" for j in range(k)]; index = JOBS.index(job)
    from validation.irt_3pl4pl_root_cause import BOUNDS, irf_rmse      # p1 definitions, unchanged
    parts = {q: [] for q in ("a", "b", "c", "d")}; converged = refused = 0; irfs = []; on_bound = []; fit_rmse = []
    for r in range(reps):
        x = simulate(model, n, k, [SEED, index, r], variant)
        fit = fit_python_mirt(x, model=model, estimation=estimation, n_quadpts=61,
                              common_discrimination=variant == "common_a").fit
        converged += bool(fit.converged)
        coef = fit.coef(); short = {"discrimination": "a", "difficulty": "b", "guessing": "c", "upper": "d"}
        est = {short[c]: coef[c].to_numpy(float) for c in coef.columns if c in short}
        if model in ("3PL", "4PL"):
            free = "abc" + ("d" if model == "4PL" else "")
            if model == "3PL":
                est["d"] = np.ones(k)
            tru = {**p, "d": np.asarray(p["d"]) if model == "4PL" else np.ones(k)}
            irfs.append(irf_rmse(est, tru))
            on_bound.append(sum(int(((est[q] - BOUNDS[q][0] <= 1e-3) | (BOUNDS[q][1] - est[q] <= 1e-3)).sum())
                                for q in free) / (k * len(free)))
            fit_rmse.append({q: float(np.sqrt(np.mean((est[q] - np.asarray(tru[q])) ** 2))) for q in free})
        table, _ = parameter_inference(fit, model, names)
        if table is None:
            refused += 1; continue
        for q in parts:
            part = table[table.Parametre == q]
            if part.empty:
                continue
            j = part.Madde.str[1:].astype(int).to_numpy(); t = np.asarray(p[q])[j]
            e = part.Kestirim.to_numpy(); low = part["GA %95 alt"].to_numpy(); high = part["GA %95 üst"].to_numpy()
            if q == "a" and variant == "common_a":
                e, low, high, t = e[:1], low[:1], high[:1], t[:1]
            parts[q].append(np.column_stack([e - t, np.isnan(low), (low <= t) & (t <= high)]))
    out = {"cell": label, "N": n, "k": k, "estimation": estimation, "n_quadpts": 61, "replications": reps,
           "converged": converged, "se_refused": refused}
    if irfs:
        out.update(irf_rmse=float(np.mean(irfs)), at_bound_rate=float(np.mean(on_bound)),
                   p1_rmse={q: float(np.mean([f[q] for f in fit_rmse])) for q in fit_rmse[0]})
    for q, rows in parts.items():
        if rows:
            arr = np.vstack(rows); shown = arr[arr[:, 1] == 0]
            out[q] = {"intervals": int(len(arr)), "withheld_rate": float(arr[:, 1].mean()),
                      "mean_bias": float(arr[:, 0].mean()), "rmse": float(np.sqrt((arr[:, 0] ** 2).mean())),
                      "coverage": float(shown[:, 2].mean()) if len(shown) else float("nan"),
                      **coverage_precision(rows, reps, eligible_column=1)}
    return out


def judge(r):
    from validation.binary_certification.coverage import judge as g3
    if r["cell"] in ("3PL MAP", "4PL MAP"):
        checks = {"convergence": r["converged"] / r["replications"] >= 0.99,
                  "irf_rmse": r["irf_rmse"] <= P1["irf_rmse"], "at_bound_rate": r["at_bound_rate"] <= P1["at_bound_rate"]}
        for q, v in r["p1_rmse"].items():
            checks[f"{q}_rmse"] = v <= (P1["rmse_a_b"] if q in "ab" else P1["rmse_c_d"])
        if r["cell"] == "3PL MAP":                    # G3 for 3PL; 4PL CI is outside its certified scope
            full = g3(r)
            checks.update({key: v for key, v in full.items() if key.endswith(("coverage", "withheld")) or key == "se_refused"})
        return checks
    return g3(r)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--output", required=True)
    a = ap.parse_args(argv); start = time.time()
    order = sorted(JOBS, key=lambda j: -j[1] * j[2] * j[3] * (3 if "PL MAP" in j[0] else 1))
    with ProcessPoolExecutor(a.workers) as pool:
        done = dict(zip(order, pool.map(_cell, order)))
    results = [done[j] for j in JOBS]
    for r in results:
        r["checks"] = judge(r); r["pass"] = all(r["checks"].values())
    import mirt
    ev = {"protocol": "validation/binary_certification/PROTOCOL.json amendment_A2", "results": results,
          "pass_by_label": {lab: all(r["pass"] for r in results if r["cell"] == lab) for lab in dict.fromkeys(j[0] for j in JOBS)},
          "wall_seconds": round(time.time() - start, 1),
          "versions": {"mirt": mirt.__version__, "numpy": np.__version__, "python": platform.python_version()}}
    Path(a.output).write_text(json.dumps(ev, indent=1, allow_nan=True) + "\n")
    print(json.dumps(ev["pass_by_label"], indent=1))


if __name__ == "__main__":
    main()
