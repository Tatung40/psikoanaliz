"""Python-side inputs for scripts/mirt_irt_b10_reference.R (B10). Run from the repository root.

Writes tests/fixtures/irt_b10/py_items_<case>.csv (production fit_irt a, b; GH 41 points),
py_persons_<case>.csv (Warm WLE with those parameters) and py_dif_anchors_<case>.csv
(purified anchors from core.irt_dif, production quadrature).
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.irt import fit_irt
from core.irt_mg import fit_single, Design
from core.irt_q3 import wle
from core.irt_mg import as_matrix
from core.irt_dif import dif_analysis

F = Path("tests/fixtures/irt_b10")
SINGLE = {"irt1pl_ok": "1PL", "irt2pl_ok": "2PL", "irt2pl_ld": "2PL", "irt2pl_missing": "2PL"}
DIF = {"dif1pl_uniform": "1PL", "dif2pl_mixed": "2PL", "dif2pl_none": "2PL"}

if __name__ == "__main__":
    for case, model in SINGLE.items():
        d = pd.read_csv(F / f"data_{case}.csv")
        if d.isna().any().any():                      # fit_irt needs complete data: use the B10 MML
            f = fit_single(d, model, information=False)
            a, b = Design(model, ["anchor"] * d.shape[1], 1).unpack(f["params"], 0)
        else:
            r = fit_irt(d, model=model, points=41, max_iter=500)
            a, b = r["items"]["a"].to_numpy(), r["items"]["b"].to_numpy()
        pd.DataFrame({"Madde": d.columns, "a": a, "b": b}).to_csv(F / f"py_items_{case}.csv", index=False,
                                                                 float_format="%.17g")
        x, obs = as_matrix(d)
        pd.DataFrame({"theta": wle(x, obs, a, b)[0]}).to_csv(F / f"py_persons_{case}.csv", index=False,
                                                             float_format="%.17g")
    for case, model in DIF.items():
        d = pd.read_csv(F / f"data_{case}.csv")
        out = dif_analysis(d.drop(columns="Grup"), d["Grup"], "R", "F", model=model)
        pd.DataFrame({"anchor": out["anchors"]}).to_csv(F / f"py_dif_anchors_{case}.csv", index=False)
        print(case, out["anchors"])
