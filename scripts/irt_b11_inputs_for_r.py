"""Python-side inputs for scripts/mirt_irt_b11_reference.R (B11). Run from the repository root.

For each single-group case writes py_items_<case>.csv (a, b, c, d of the bounded MML fit,
core.irt_mg34, Gauss-Hermite 41) and py_persons_<case>.csv (EAP and 3PL/4PL Warm WLE with
those parameters); for each DIF case py_dif_anchors_<case>.csv (anchors used for the
final multigroup comparison).
"""
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.irt_mg import as_matrix
from core.irt_mg34 import fit_single, Design34
from core.irt_q3 import wle_general, eap

F = Path("tests/fixtures/irt_b11")
SINGLE = {"irt3pl_ok": "3PL", "irt3pl_ld": "3PL", "irt3pl_missing": "3PL", "irt3pl_easy": "3PL",
          "irt4pl_ok": "4PL", "irt4pl_ld": "4PL"}
ANCHORS = {"dif3pl": ["M1", "M2", "M4", "M5", "M6", "M7", "M9", "M11", "M12"],
           "dif3pl_none": [f"M{i}" for i in range(1, 13)],
           "dif4pl": ["M1", "M2", "M4", "M5", "M6", "M7", "M9", "M10", "M11", "M12"],
           "dif4pl_none": [f"M{i}" for i in range(1, 13)]}

if __name__ == "__main__":
    only = set(sys.argv[1:])
    for case, model in SINGLE.items():
        if only and model not in only:
            continue
        d = pd.read_csv(F / f"data_{case}.csv")
        f = fit_single(d, model, information=False)
        a, b, c, dd = Design34(model, ["anchor"] * d.shape[1], 1).unpack(f["params"], 0)
        pd.DataFrame({"Madde": d.columns, "a": a, "b": b, "c": c, "d": dd}).to_csv(
            F / f"py_items_{case}.csv", index=False, float_format="%.17g")
        x, obs = as_matrix(d)
        w, _, edge, _ = wle_general(x, obs, a, b, c, dd)          # NaN where the WLE root is not unique
        pd.DataFrame({"wle": w, "wle_edge": edge.astype(int), "eap": eap(x, obs, a, b, 41, c=c, d=dd)[0]}).to_csv(
            F / f"py_persons_{case}.csv", index=False, float_format="%.17g")
    for case, anchors in ANCHORS.items():
        if only and not any(m.lower() in case for m in only):
            continue
        pd.DataFrame({"anchor": anchors}).to_csv(F / f"py_dif_anchors_{case}.csv", index=False)
    print("inputs written")
