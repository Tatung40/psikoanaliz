"""Amendment B2: 4PL G1 with R mirt plus an independent numerical reference (PROTOCOL.json amendment_B).

    python validation/binary_certification/b2_4pl.py prepare --out DIR      # fixtures, Python fits, numpy reference
    Rscript validation/binary_certification/reference.R DIR/manifest.json DIR   (independent R starts, A3-A5)
    Rscript validation/binary_certification/b2_checks.R DIR/manifest.json DIR  (R log-posterior at both; matched init)
    python validation/binary_certification/b2_4pl.py compare --out DIR --evidence FILE
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from validation.binary_certification.common import Objective  # noqa: E402
from validation.binary_certification.r_equivalence import G1, SIM, cases, load, python_fit  # noqa: E402

C = G1["criteria"]
FRESH = ["sim(N=3000, k=12, seed=404)", "sim(N=3000, k=30, seed=405)", "sim(N=10000, k=12, seed=406)", "sim(N=10000, k=30, seed=407)"]


def b2_cases():
    registered = [c for c in cases() if c["model"] == "4PL"]
    fresh = [dict(model="4PL", variant=None, fixture=fx, estimation="MAP",
                  id="4PL__fresh_" + "_".join(SIM.fullmatch(fx).groups())) for fx in FRESH]
    return registered + fresh


def prepare(out):
    from validation.binary_certification.reference_numpy import fit
    out.mkdir(parents=True, exist_ok=True); manifest = []
    for case in b2_cases():
        x = load(case); path = out / f"{case['id']}.csv"
        pd.DataFrame(x, columns=[f"V{j + 1}" for j in range(x.shape[1])]).to_csv(path, index=False)
        case.update(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), shape=list(x.shape))
        (out / f"{case['id']}.python.json").write_text(json.dumps(python_fit(x, case)))
        ref, info = fit(x, "4PL")
        (out / f"{case['id']}.numpy.json").write_text(json.dumps({"par": {q: v.tolist() for q, v in ref.items()}, **info}))
        manifest.append(case); print("prepared", case["id"], info["converged"], flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


def _agree(objective, P, Q):
    diff = {q: float(np.max(np.abs(P[q] - Q[q]))) for q in P}
    fp, fq = objective.value(P), objective.value(Q)
    ok = (max(diff["a"], diff["b"]) <= C["max_abs_diff_a_b"] and max(diff["c"], diff["d"]) <= C["max_abs_diff_c_d"]
          and abs(fp - fq) <= C["abs_objective_diff"] and fp >= fq - C["python_not_worse_than_R_by_more_than"])
    return bool(ok), diff, fp - fq


def compare(out, evidence):
    rows = []
    for case in json.loads((out / "manifest.json").read_text()):
        x = pd.read_csv(case["path"]).to_numpy(int); obj = Objective(x, "4PL", map_priors=True)
        load_par = lambda name: {q: np.asarray(v, float) for q, v in json.loads((out / name).read_text())["par"].items()}
        P = load_par(f"{case['id']}.python.json"); row = {"id": case["id"], "fixture": case["fixture"], "sha256": case["sha256"]}
        try:
            r = json.loads((out / f"{case['id']}.R.json").read_text())
            original, d_r, df_r = _agree(obj, P, {q: np.asarray(r["par"][q], float) for q in P})
            original = original and bool(r["converged"])
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            original, d_r, df_r = False, None, None
        ref = json.loads((out / f"{case['id']}.numpy.json").read_text())
        i_ok, d_ref, df_ref = _agree(obj, P, load_par(f"{case['id']}.numpy.json"))
        i_ok = i_ok and ref["converged"]
        try:
            b = json.loads((out / f"{case['id']}.Rb2.json").read_text())
            ii_ok = b["R_logpost_at_python"] >= b["R_logpost_at_R"]
            iii_ok, d_m, _ = _agree(obj, P, {q: np.asarray(b["matched_par"][q], float) for q in P})
            iii_ok = bool(max(d_m["a"], d_m["b"]) <= C["max_abs_diff_a_b"] and max(d_m["c"], d_m["d"]) <= C["max_abs_diff_c_d"])
            row.update(R_logpost_at_python=b["R_logpost_at_python"], R_logpost_at_R=b["R_logpost_at_R"], matched_max_abs_diff=d_m)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            ii_ok = iii_ok = False
        row.update(original_R_pass=original, R_max_abs_diff=d_r, R_objective_gap=df_r,
                   reference_pass=i_ok, reference_max_abs_diff=d_ref, reference_objective_gap=df_ref,
                   reference_iterations=ref["iterations"], R_prefers_python=bool(ii_ok), R_matched_stationary=bool(iii_ok),
                   pass_=bool(original or (i_ok and ii_ok and iii_ok)))
        rows.append(row); print(case["id"], "PASS" if row["pass_"] else "FAIL", "orig", original, "ref", i_ok, "R-prefers", ii_ok, "matched", iii_ok)
    res = {"protocol": "PROTOCOL.json amendment_B B2", "cases": rows, "all_pass": all(r["pass_"] for r in rows) and len(rows) == 7}
    Path(evidence).write_text(json.dumps(res, indent=1, allow_nan=True) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("step", choices=("prepare", "compare"))
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--evidence")
    a = ap.parse_args()
    prepare(a.out) if a.step == "prepare" else compare(a.out, a.evidence)
