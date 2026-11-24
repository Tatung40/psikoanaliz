"""Amendment C1: per-job outputs (fixture x R start) -> the unchanged B2 rule per fixture.

    python validation/binary_certification/c1_combine.py --out DIR --start default|nested --evidence FILE   (per job)
    python validation/binary_certification/c1_combine.py --combine JOB1.json JOB2.json ... --evidence FILE
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from validation.binary_certification.b2_4pl import C, _agree  # noqa: E402
from validation.binary_certification.common import Objective  # noqa: E402


def job(out, start):
    rows = []
    for case in json.loads((out / "manifest.json").read_text()):
        x = pd.read_csv(case["path"]).to_numpy(int); obj = Objective(x, "4PL", map_priors=True)
        par = lambda d: {q: np.asarray(v, float) for q, v in d.items()}
        P = par(json.loads((out / f"{case['id']}.python.json").read_text())["par"])
        ref = json.loads((out / f"{case['id']}.numpy.json").read_text())
        row = {"id": case["id"], "sha256": case["sha256"], "start": start, "reference_converged": ref["converged"]}
        row["reference_pass"], row["reference_max_abs_diff"], row["reference_objective_gap"] = _agree(obj, P, par(ref["par"]))
        rp, bp = out / f"{case['id']}.R.json", out / f"{case['id']}.Rb2.json"
        if rp.exists():
            r = json.loads(rp.read_text())
            row["R_finished"] = True
            row["R_original_pass"], row["R_max_abs_diff"], row["R_objective_gap"] = _agree(obj, P, par(r["par"]))
            row["R_original_pass"] = row["R_original_pass"] and bool(r["converged"])
        else:
            row["R_finished"] = False
        if bp.exists():
            b = json.loads(bp.read_text())
            row["R_logpost_at_python"], row["R_logpost_at_R"] = b.get("R_logpost_at_python"), b.get("R_logpost_at_R")
            if b.get("matched_par"):
                d = {q: float(np.max(np.abs(P[q] - np.asarray(b["matched_par"][q], float)))) for q in P}
                row["matched_max_abs_diff"] = d
                row["matched_pass"] = max(d["a"], d["b"]) <= C["max_abs_diff_a_b"] and max(d["c"], d["d"]) <= C["max_abs_diff_c_d"]
        rows.append(row)
    return rows


def combine(rows):
    out = {}
    for fid in sorted({r["id"] for r in rows}):
        rs = [r for r in rows if r["id"] == fid]
        done = [r for r in rs if r.get("R_finished") and r.get("R_logpost_at_R") is not None]
        best = max(done, key=lambda r: r["R_logpost_at_R"]) if done else None
        ref_ok = all(r["reference_pass"] and r["reference_converged"] for r in rs)
        matched = [r.get("matched_pass") for r in rs if "matched_pass" in r]
        original = bool(best and best["R_original_pass"])
        prefers = bool(best and all(r["R_logpost_at_python"] >= r["R_logpost_at_R"] for r in done))
        out[fid] = {"starts_finished": [r["start"] for r in done], "best_start": best and best["start"],
                    "original_R_pass": original, "reference_pass": ref_ok, "R_prefers_python": prefers,
                    "R_matched_stationary": bool(matched and all(matched)),
                    "pass_": original or (ref_ok and prefers and bool(matched and all(matched)))}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path); ap.add_argument("--start")
    ap.add_argument("--combine", nargs="*"); ap.add_argument("--evidence", required=True)
    a = ap.parse_args()
    if a.combine:
        rows = [r for f in a.combine for r in json.loads(Path(f).read_text())]
        res = {"protocol": "PROTOCOL.json amendment_C C1 (B2 rule, 900 s per start)", "jobs": rows, "fixtures": combine(rows)}
        res["all_pass"] = len(res["fixtures"]) == 3 and all(v["pass_"] for v in res["fixtures"].values())
    else:
        res = job(a.out, a.start)
    Path(a.evidence).write_text(json.dumps(res, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o)) + "\n")
    print(json.dumps(res if a.combine else [{k: r.get(k) for k in ("id", "start", "R_finished", "R_original_pass", "R_logpost_at_python", "R_logpost_at_R", "matched_pass")} for r in res], indent=1, default=str))
