"""Python side of the free 4PL comparison with R mirt (B5).

Fits the B5 fixtures with fit_irt (GH-21, tolerance 1e-9, max_iter 5000) from
the three multistart shifts (0, -1, +1), then maximizes the same marginal
log-likelihood inside the same box with L-BFGS-B (validated logLik and analytic
score) from every Python and mirt solution, to locate the bounded optimum
independently of either EM. Writes tests/fixtures/mirt/python_free4pl_results.csv.
Run from the repository root after scripts/mirt_free_4pl.R:
    python scripts/python_free_4pl.py
Expect several minutes on a laptop; not part of the default test suite.
"""
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.irt import fit_irt
from core.irt_hessian import marginal_score
from core.irt_quadrature_validation import marginal_loglik_at_fixed_parameters
from core.irt_stationarity import stationarity_diagnostics

FIX = Path("tests/fixtures/mirt")
BOX = [(.15, 4.), (-4., 4.), (0., .35), (.65, 1.)]   # fit_irt defaults for 4PL


def mirt_solution(results, case, run, items):
    e = results[(results.case == case) & (results.run == run) & (results.quantity == "estimate")]
    e = e.pivot(index="row", columns="col", values="value").loc[items]
    return pd.DataFrame({"Madde": items, "a": e.a1.values, "b": -e.d.values / e.a1.values,
                         "c": e.g.values, "d": e.u.values})


def polish(x, table):
    k = len(table)
    def tab(v):
        t = pd.DataFrame(v.reshape(k, 4), columns=list("abcd")); t.insert(0, "Madde", table.Madde.values)
        return t
    lo, hi = np.array([b[0] for b in BOX] * k), np.array([b[1] for b in BOX] * k)
    v0 = np.clip(table[list("abcd")].to_numpy(float).reshape(-1), lo, hi)
    n = len(x)
    out = minimize(lambda v: -marginal_loglik_at_fixed_parameters(x, tab(v), points=21) / n, v0,
                   jac=lambda v: -marginal_score(x, tab(v), "4PL", points=21) / n,
                   method="L-BFGS-B", bounds=BOX * k,
                   options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-11})
    return tab(out.x), int(out.nit)


def main():
    mirt = pd.read_csv(FIX / "mirt_free4pl_results.csv", keep_default_na=False, na_values=["NA"])
    rows = []
    def record(case, source, run, table, extra):
        _, kkt = stationarity_diagnostics(x, table, "4PL", points=21)
        base = {"case": case, "source": source, "run": run,
                "loglik": marginal_loglik_at_fixed_parameters(x, table, points=21),
                "max_kkt_per_person": kkt["max_KKT_violation"], **extra}
        for _, it in table.iterrows():
            for p in "abcd":
                rows.append({**base, "item": it.Madde, "param": p, "value": float(it[p])})
    for case in pd.read_csv(FIX / "cases_free4pl.csv").case:
        x = pd.read_csv(FIX / f"data_{case}.csv")
        items = list(x.columns)
        for shift in (0., -1., 1.):
            t0 = time.time()
            fit = fit_irt(x, "4PL", points=21, max_iter=5000, tolerance=1e-9, initial_shift=shift)
            record(case, "python_em", f"s{shift:g}", fit["items"][["Madde", "a", "b", "c", "d"]],
                   {"converged": fit["converged"], "iterations": fit["iterations"],
                    "status": fit["convergence"]["status"], "seconds": round(time.time() - t0, 1)})
            polished, nit = polish(x, fit["items"])
            record(case, "polished_from_python", f"s{shift:g}", polished, {"iterations": nit})
        polished, nit = polish(x, mirt_solution(mirt, case, "box_s0", items))
        record(case, "polished_from_mirt", "box_s0", polished, {"iterations": nit})
    pd.DataFrame(rows).to_csv(FIX / "python_free4pl_results.csv", index=False, float_format="%.12g")
    print("written", len(rows), "rows")


if __name__ == "__main__":
    main()
