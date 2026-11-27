import json, sys
import pandas as pd
from core.dif_classical import dif_screen
d = pd.read_csv("tests/fixtures/irt_b10/data_dif2pl_mixed.csv")
items = [c for c in d.columns if c not in ("Grup", "ID", "id")]
out = {}
for m in ("MH", "LR"):
    res = dif_screen(d[items], d["Grup"].to_numpy(), "R", "F", method=m, purify=False, adjust="none")
    f = res[0] if isinstance(res, tuple) else res
    out[m] = f.to_dict(orient="list")
out["items"] = items
json.dump(out, open(sys.argv[1], "w"), default=float)
