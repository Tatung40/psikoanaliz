import json, sys
import numpy as np
P = json.load(open(sys.argv[1])); R = json.load(open(sys.argv[2])); tol = json.load(open("validation/dif_difr/PROTOCOL.json"))["tolerances"]
def cmp(name, a, b, t):
    a, b = np.asarray(a, float), np.asarray(b, float); e = float(np.nanmax(np.abs(a - b))); return dict(statistic=name, max_abs_error=e, tolerance=t, agree=bool(e <= t and np.isfinite(a).all()))
rows = [cmp("MH chi2", P["MH"]["chi2"], R["mh_chi2"], tol["MH chi2"]), cmp("alpha_MH", P["MH"]["alpha_mh"], R["mh_alpha"], tol["alpha_MH"]),
        cmp("MH p", P["MH"]["p"], R["mh_p"], tol["MH p"]), cmp("LR omnibus", P["LR"]["lr_omnibus"], R["lr"], tol["LR omnibus (2 df)"]),
        cmp("LR p", P["LR"]["p_omnibus"], R["lr_p"], tol["LR p"])]
rep = {"difR": R["difR"], "rows": rows, "agree_all": all(r["agree"] for r in rows), "status": "NUMERICAL AGREEMENT ONLY"}
print("DIFR " + json.dumps(rep))
