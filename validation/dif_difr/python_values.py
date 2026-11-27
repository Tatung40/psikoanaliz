"""Production values on 3 registered fixtures: MH (chi2, alpha, ETS delta, p, BH p), LR (omnibus, uniform, p, Nagelkerke dR2), SIBTEST."""
import json, sys
import numpy as np, pandas as pd
import mirt
from core.dif_classical import dif_screen
FIX = ("dif2pl_mixed", "dif2pl_none"); out = {}   # dif1pl_uniform has missing responses: observed-score methods refuse by design
for f in FIX:
    d = pd.read_csv(f"tests/fixtures/irt_b10/data_{f}.csv"); items = [c for c in d.columns if c not in ("Grup", "ID", "id")]
    anchors = pd.read_csv(f"tests/fixtures/irt_b10/py_dif_anchors_{f}.csv")["anchor"].tolist(); g = d["Grup"].to_numpy()
    mh = dif_screen(d[items], g, "R", "F", method="MH", purify=False, adjust="BH")[0]
    lr = dif_screen(d[items], g, "R", "F", method="LR", purify=False, adjust="BH")[0]
    mhp = dif_screen(d[items], g, "R", "F", method="MH", purify=True, adjust="none")[0]
    x = d[items].to_numpy(int); gf = (g == "F").astype(int); A = [items.index(a) for a in anchors]
    sib = [mirt.sibtest(x, gf, suspect_items=[j], matching_items=A, focal_group=1) for j in range(len(items)) if j not in A]
    out[f] = dict(items=items, anchors=anchors, mh_chi2=mh["chi2"].tolist(), mh_alpha=mh["alpha_mh"].tolist(), mh_delta=mh["delta"].tolist(),
                  mh_p=mh["p"].tolist(), mh_p_bh=mh["p_adj"].tolist(), mh_purified_chi2=mhp["chi2"].tolist(),
                  lr_omnibus=lr["lr_omnibus"].tolist(), lr_uniform=lr["lr_uniform"].tolist(), lr_p=lr["p_omnibus"].tolist(),
                  lr_dr2=lr["delta_r2"].tolist(), sib_items=[items[j] for j in range(len(items)) if j not in A],
                  sib_beta=[float(r["beta"]) for r in sib], sib_chi2=[float(r["chi2"]) for r in sib], sib_p=[float(r["p_value"]) for r in sib])
json.dump(out, open(sys.argv[1], "w"), default=float)
