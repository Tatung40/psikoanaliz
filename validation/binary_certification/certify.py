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
    """Status per PROTOCOL.json and its amendments: G1 from r_equivalence.json (harness fixed by A3); G2/G3 of all four
    models from amendment A2 (corrected 61-node default) at all four corners. Earlier evidence stays reported only."""
    g1, a2 = _load("binary_certification/r_equivalence.json"), _load("binary_certification/amendment_a2.json")
    gates = {m: {} for m in ("1PL", "2PL", "3PL", "4PL")}
    for m in gates:
        cases = [c for c in (g1 or {}).get("cases", []) if c["model"] == m]
        want = len(PROTOCOL["gates"]["G1_R_equivalence"]["fixtures"][m]) * (2 if m == "1PL" else 1)
        gates[m]["G1_R_equivalence"] = {"pass": len(cases) == want and all(c["pass_"] for c in cases),
                                        "cases": {c["id"]: c["pass_"] for c in cases}, "expected_cases": want}
    labels = {"1PL": ("1PL fixed_a", "1PL common_a"), "2PL": ("2PL",), "3PL": ("3PL MAP",), "4PL": ("4PL MAP",)}
    for m, labs in labels.items():
        cells = {f"{r['cell']} N{r['N']} k{r['k']}": r for r in (a2 or {}).get("results", []) if r["cell"] in labs}
        want = 4 * len(labs)
        g2 = {key: all(v for c, v in r["checks"].items() if not c.endswith(("coverage", "withheld")) and c != "se_refused")
              for key, r in cells.items()}
        gates[m]["G2_recovery"] = {"pass": len(g2) == want and all(g2.values()), "cells": g2, "source": "amendment_a2.json"}
        if m == "4PL":
            gates[m]["G3_inference"] = {"pass": None, "note": PROTOCOL["scope"]["4PL"]["excluded_outputs"]}
        else:
            g3 = {key: all(v for c, v in r["checks"].items() if c.endswith(("coverage", "withheld")) or c in ("se_refused", "convergence"))
                  for key, r in cells.items()}
            gates[m]["G3_inference"] = {"pass": len(g3) == want and all(g3.values()), "cells": g3, "source": "amendment_a2.json",
                                        "failures": {key: [c for c, v in cells[key]["checks"].items() if not v]
                                                     for key, ok in g3.items() if not ok}}
    status = {}
    for m, g in gates.items():
        ok = all(v["pass"] for v in g.values() if v["pass"] is not None)
        status[m] = {"status": "CERTIFIED" if ok else "NOT CERTIFIED", "scope": PROTOCOL["scope"][m],
                     "failed_gates": [name for name, v in g.items() if v["pass"] is False], "gates": g}
    return {"protocol": "validation/binary_certification/PROTOCOL.json (amendments A1-A3)", "models": status,
            "superseded_evidence_reported": ["p1_3pl4pl", "p6_binary_se", "p7_binary34_se", "binary_certification/coverage.json",
                                             "binary_certification/amendment_a1.json", "binary_certification/r_equivalence_run1_3bd40ad.json"]}


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
