"""Infit/Outfit of library_itemfit equal independent EAP-plug-in residual mean squares (all binary models):
outfit_j = mean_i (x - P)^2 / (P (1 - P)), infit_j = sum_i (x - P)^2 / sum_i P (1 - P), P = P_j(EAP theta_i).
Computational identity only; no null calibration (the library reports no Infit/Outfit p-values)."""
import json, sys
from pathlib import Path
import numpy as np, mirt
from scipy.special import expit
from validation.binary_certification.common import simulate
from core.irt_python_mirt_adapter import fit_python_mirt
from core.irt_library_diagnostics import library_itemfit
rows = []
for model, est, n in (("Rasch", "ML", 1000), ("1PL", "ML", 1000), ("2PL", "ML", 1000), ("3PL", "MAP", 3000), ("4PL", "MAP", 3000)):
    x = simulate("1PL" if model == "Rasch" else model, n, 12, 41)
    f = fit_python_mirt(x, model=model, estimation=est, n_quadpts=61).fit; co = f.coef(); k = x.shape[1]
    a = co["discrimination"].to_numpy(float) if "discrimination" in co else np.ones(k); b = co["difficulty"].to_numpy(float)
    c = co["guessing"].to_numpy(float) if "guessing" in co else 0.0; d = co["upper"].to_numpy(float) if "upper" in co else 1.0
    th = np.asarray(mirt.fscores(f, x, method="EAP").theta).reshape(-1); P = c + (d - c) * expit(a * (th[:, None] - b))
    R2, V = (x - P) ** 2, P * (1 - P); lib = library_itemfit(f, x, model=model)
    dout = float(np.abs((R2 / V).mean(0) - lib.outfit.to_numpy()).max()); dinf = float(np.abs(R2.sum(0) / V.sum(0) - lib.infit.to_numpy()).max())
    rows.append(dict(model=model, estimation=est, outfit_max_abs_diff=dout, infit_max_abs_diff=dinf, passed=dout <= 1e-10 and dinf <= 1e-10))
    print(rows[-1])
Path(sys.argv[1]).write_text(json.dumps({"cases": rows, "all_pass": all(r["passed"] for r in rows)}, indent=1) + "\n")
