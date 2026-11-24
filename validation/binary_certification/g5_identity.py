"""Amendment C2 G5: the application's IRF and item/test information equal an independent closed form from the fitted
parameters, P = c + (d - c) / (1 + exp(-a (t - b))), I = a^2 (P - c)^2 (d - P)^2 / ((d - c)^2 P (1 - P)); tolerance 1e-10.

    PYTHONPATH=. python validation/binary_certification/g5_identity.py --output validation/evidence/binary_certification/g5_identity.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from validation.binary_certification.r_equivalence import cases, load

TOL = 1e-10


def closed_form(a, b, c, d, t):
    e = np.exp(-a * (t[:, None] - b)); P = c + (d - c) / (1 + e)
    return P, a ** 2 * (P - c) ** 2 * (d - P) ** 2 / ((d - c) ** 2 * P * (1 - P))


def check(case):
    from core.irt_library_parameter_mapping import curves_from_normalized_table
    from core.irt_python_mirt_adapter import fit_python_mirt
    from core.irt_python_mirt_parameters import python_mirt_binary_parameters
    from core.irt_python_mirt_report import build_report
    x = load(case); k = x.shape[1]; names = [f"V{j + 1}" for j in range(k)]; m = case["model"]
    out = fit_python_mirt(x, model=m, estimation="MAP", n_quadpts=61)
    co = out.fit.coef(); a, b, c = (co[q].to_numpy(float) for q in ("discrimination", "difficulty", "guessing"))
    d = co["upper"].to_numpy(float) if m == "4PL" else np.ones(k)
    rep = build_report(out, model=m, items=names)                                     # library screen
    t = np.sort(rep.curves.Theta.unique()); P, I = closed_form(a, b, c, d, t)
    cur = rep.curves[rep.curves.Kategori.astype(str) == "1"].pivot(index="Theta", columns="Madde", values="Olasılık")[names].to_numpy()
    info = rep.information.set_index("θ").loc[t]
    lib_irf = float(np.abs(cur - P).max()); lib_info = float(np.abs(info[[f"I_{n}" for n in names]].to_numpy() - I).max())
    lib_test = float(np.abs(info["Test bilgisi"].to_numpy() - I.sum(1)).max())
    table = python_mirt_binary_parameters(out.fit, model=m, item_names=names)          # binary route
    grid = np.linspace(-4, 4, 81); P2, I2 = closed_form(a, b, c, d, grid)
    curves = curves_from_normalized_table(table[["Madde", "a", "b", "c", "d"]], grid)
    par = float(np.abs(table[["a", "b", "c", "d"]].to_numpy() - np.column_stack([a, b, c, d])).max())
    rt_irf = float(np.abs(curves["probability"] - P2).max()); rt_info = float(np.abs(curves["information"] - I2).max())
    rt_test = float(np.abs(curves["test_information"] - I2.sum(1)).max())
    diffs = dict(library_irf=lib_irf, library_item_information=lib_info, library_test_information=lib_test,
                 route_parameters=par, route_irf=rt_irf, route_item_information=rt_info, route_test_information=rt_test)
    return {"id": case["id"], "model": m, "max_abs_diff": diffs, "pass": all(v <= TOL for v in diffs.values())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--output", required=True); a = ap.parse_args()
    rows = [check(c) for c in cases() if c["model"] in ("3PL", "4PL")]
    res = {"protocol": "PROTOCOL.json amendment_C C2 G5", "tolerance": TOL, "cases": rows,
           "pass_by_model": {m: all(r["pass"] for r in rows if r["model"] == m) for m in ("3PL", "4PL")}}
    Path(a.output).write_text(json.dumps(res, indent=1) + "\n")
    for r in rows: print(r["id"], r["pass"], {k: f"{v:.1e}" for k, v in r["max_abs_diff"].items()})
