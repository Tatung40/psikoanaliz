"""Apply PROTOCOL.json status rules to the reused and new evidence; write CERTIFICATION.json.

    python validation/binary_certification/certify.py                 # (re)write the decision
    python validation/binary_certification/certify.py --check-g1 FILE  # CI: recomputed G1 must match committed evidence
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "validation/evidence"
OUT = EV / "binary_certification/CERTIFICATION.json"
PROTOCOL = json.loads((Path(__file__).with_name("PROTOCOL.json")).read_text())


def _load(rel):
    path = EV / rel
    return json.loads(path.read_text()) if path.exists() else None


def decide():
    g1, cov = _load("binary_certification/r_equivalence.json"), _load("binary_certification/coverage.json")
    p1, p6, p7 = _load("p1_3pl4pl/irt_3pl4pl_root_cause.json"), _load("p6_binary_se/coverage.json"), _load("p7_binary34_se/coverage.json")
    gates = {m: {} for m in ("1PL", "2PL", "3PL", "4PL")}
    for m in gates:                                                             # G1
        cases = [c for c in (g1 or {}).get("cases", []) if c["model"] == m]
        want = len(PROTOCOL["gates"]["G1_R_equivalence"]["fixtures"][m]) * (2 if m == "1PL" else 1)
        gates[m]["G1_R_equivalence"] = {"pass": len(cases) == want and all(c["pass_"] for c in cases),
                                        "cases": {c["id"]: c["pass_"] for c in cases}, "expected_cases": want}
    new = {(r["cell"], r["N"], r["k"]): r for r in (cov or {}).get("results", [])}
    def cells(label):
        return [(n, k) for n, k in PROTOCOL["gates"]["G3_inference"]["new_cells"].get(label, [])]
    # 1PL (both variants) and 2PL: reused p6 corner (1000, 12) + new corners; recovery and inference per cell.
    for m, labels in (("1PL", ("1PL fixed_a", "1PL common_a")), ("2PL", ("2PL",))):
        detail = {}
        for label in labels:
            for n, k in cells(label):
                r = new.get((label, n, k)); detail[f"{label} N{n} k{k}"] = bool(r and r["pass"])
            if label != "1PL common_a":
                key = "1PL" if m == "1PL" else "2PL"
                detail[f"{label} N1000 k12 (p6)"] = bool(p6 and all(v for c, v in p6["checks"].items() if c.startswith(key + "_")))
        gates[m]["G2_G3_recovery_inference"] = {"pass": bool(detail) and all(detail.values()), "cells": detail}
    for m in ("3PL", "4PL"):                                                    # G2 reused p1 corners (MAP)
        corners = {f"N{n} k{k}": bool(p1 and p1["summary"].get(f"{m}|N{n}|k{k}|MAP|normal", {}).get("meets_preregistered_criteria"))
                   for n in (3000, 10000) for k in (12, 30)}
        gates[m]["G2_recovery"] = {"pass": all(corners.values()), "cells": corners, "source": "p1_3pl4pl"}
    detail = {f"3PL MAP N{n} k{k}": bool(new.get(("3PL MAP", n, k), {}).get("pass")) for n, k in cells("3PL MAP")}
    p7cell = next((r for r in (p7 or {}).get("results", []) if (r["model"], r["estimation"], r["N"], r["items"]) == ("3PL", "MAP", 3000, 30)), None)
    detail["3PL MAP N3000 k30 (p7)"] = bool(p7 and all(v for c, v in p7["checks"].items() if c.startswith("3PL_MAP_N3000")))
    gates["3PL"]["G3_inference"] = {"pass": all(detail.values()), "cells": detail, "p7_cell": p7cell is not None}
    gates["4PL"]["G3_inference"] = {"pass": None, "note": PROTOCOL["scope"]["4PL"]["excluded_outputs"]}
    status = {}
    for m, g in gates.items():
        ok = all(v["pass"] for v in g.values() if v["pass"] is not None)
        status[m] = {"status": "CERTIFIED" if ok else "NOT CERTIFIED", "scope": PROTOCOL["scope"][m],
                     "failed_gates": [name for name, v in g.items() if v["pass"] is False], "gates": g}
    return {"protocol": "validation/binary_certification/PROTOCOL.json", "models": status}


def check_g1(path):
    committed = _load("binary_certification/r_equivalence.json")
    fresh = json.loads(Path(path).read_text())
    if committed is None:
        sys.exit("No committed G1 evidence: commit the CI artifact r_equivalence.json first.")
    old = {c["id"]: c for c in committed["cases"]}
    for c in fresh["cases"]:
        o = old.get(c["id"])
        if o is None or o["pass_"] != c["pass_"] or o.get("sha256") != c.get("sha256"):
            sys.exit(f"G1 recomputation differs from committed evidence: {c['id']}")
        for q, v in c.get("max_abs_diff", {}).items():
            if abs(v - o["max_abs_diff"][q]) > 1e-6:
                sys.exit(f"G1 numbers drifted: {c['id']} {q}")
    if len(old) != len(fresh["cases"]):
        sys.exit("G1 case set differs from committed evidence")
    stored = json.loads(OUT.read_text())
    if stored != decide():
        sys.exit("CERTIFICATION.json is not the decision of the committed evidence")
    print("G1 recomputation matches committed evidence; decision unchanged")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--check-g1")
    a = ap.parse_args()
    if a.check_g1:
        check_g1(a.check_g1)
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(decide(), indent=1) + "\n")
        print(json.dumps({m: v["status"] for m, v in decide()["models"].items()}))
