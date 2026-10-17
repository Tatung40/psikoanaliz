"""B6 benchmark: default EM vs EM + bounded L-BFGS-B refinement (fit_irt(refine=True)).

Uses the B5 fixtures (4PL k=5/k=8, N=3000) and the B3 3PL estimation fixture;
accuracy is measured against the B5 common bounded optimum and the mirt 3PL MLE.
Each timing is repeated (default 3) on the same machine. Run from the repo root:
    python scripts/irt_refine_benchmark.py [repeats]
"""
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.irt import fit_irt
from core.irt_hessian import observed_information_se

FIX = Path("tests/fixtures/mirt")


def reference(case, items):
    if case == "3pl_est":
        r = pd.read_csv(FIX / "mirt_results.csv"); e = r[r.case == case]
        cf = e[e.quantity == "fit_irtpars"].pivot(index="row", columns="col", values="value").loc[items]
        return float(e[e.quantity == "fit_loglik"].value.iloc[0]), np.column_stack([cf.a, cf.b, cf.g])
    p = pd.read_csv(FIX / "python_free4pl_results.csv")
    q = p[(p.case == case) & (p.source == "polished_from_mirt")]
    return float(q.loglik.iloc[0]), q.pivot(index="item", columns="param", values="value").loc[items][list("abcd")].to_numpy()


def main(repeats=3):
    rows = []
    for case, model in (("4pl_free_k5", "4PL"), ("4pl_free_k8", "4PL"), ("3pl_est", "3PL")):
        x = pd.read_csv(FIX / f"data_{case}.csv")
        ll_ref, par_ref = reference(case, list(x.columns))
        cols = ["a", "b", "c"] if model == "3PL" else list("abcd")
        for label, kw in (("EM (default)", {}), ("EM + L-BFGS-B", {"refine": True})):
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter(); fit = fit_irt(x, model, points=21, **kw); times.append(time.perf_counter() - t0)
            _, reason = observed_information_se(x, fit["items"], model, points=21)
            rf = fit["refinement"]
            rows.append({"case": case, "method": label, "seconds_mean": np.mean(times), "seconds_sd": np.std(times),
                         "em_iterations": fit["iterations"], "refine_iterations": rf.get("iterations", 0),
                         "dLL_vs_reference": fit["loglik"] - ll_ref,
                         "max_abs_param_diff": float(np.abs(fit["items"][cols].to_numpy() - par_ref).max()),
                         "max_kkt": rf.get("max_kkt_final", np.nan), "boundary_params": fit["convergence"]["parameters_at_bound"],
                         "status": fit["convergence"]["status"], "se": "produced" if not reason else "refused"})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    return out


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
