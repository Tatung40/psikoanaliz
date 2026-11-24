"""Independent S-X2 (Orlando & Thissen) vs library_itemfit, items whose df shows no cell collapsing."""
import numpy as np, sys
sys.path.insert(0, "/home/claude/psikoanaliz")
from scipy.special import expit, roots_hermite
from validation.binary_certification.common import simulate
from core.irt_python_mirt_adapter import fit_python_mirt
from core.irt_library_diagnostics import library_itemfit
def lw(P):  # P: Q x k -> Q x (k+1) summed-score likelihood
    f = np.ones((P.shape[0], 1))
    for j in range(P.shape[1]):
        g = np.zeros((P.shape[0], f.shape[1] + 1)); g[:, :-1] += f * (1 - P[:, [j]]); g[:, 1:] += f * P[:, [j]]; f = g
    return f
def sx2(x, a, b, c, d, npar, pts=61):
    z, w = roots_hermite(pts); t = np.sqrt(2) * z; w = w / np.sqrt(np.pi)
    P = c + (d - c) * expit(a * (t[:, None] - b)); k = x.shape[1]; s = x.sum(1); F = lw(P); out = []
    for i in range(k):
        Fi = lw(np.delete(P, i, 1)); stat = 0
        for sc in range(1, k):
            Ns = (s == sc).sum(); O = x[s == sc, i].mean()
            E = (w * P[:, i] * Fi[:, sc - 1]).sum() / (w * F[:, sc]).sum()
            stat += Ns * (O - E) ** 2 / (E * (1 - E))
        out.append(stat)
    return np.array(out), k - 1 - npar
import json; rows=[]
for model, est, n in (("Rasch", "ML", 5000), ("1PL", "ML", 5000), ("2PL", "ML", 5000), ("3PL", "MAP", 5000), ("4PL", "MAP", 5000)):
    x = simulate("1PL" if model == "Rasch" else model, n, 10, 77)
    f = fit_python_mirt(x, model=model, estimation=est, n_quadpts=61).fit; co = f.coef(); k = 10
    a = co["discrimination"].to_numpy(float) if "discrimination" in co else np.ones(k); b = co["difficulty"].to_numpy(float)
    c = co["guessing"].to_numpy(float) if "guessing" in co else 0.0; d = co["upper"].to_numpy(float) if "upper" in co else 1.0
    lib = library_itemfit(f, x, model=model); npar = {"Rasch": 1, "1PL": 1, "2PL": 2, "3PL": 3, "4PL": 4}[model]
    mine, df = sx2(x, a, b, c, d, npar); same = lib.df.to_numpy() == df
    rel = float(np.max(np.abs(mine[same] - lib.S_X2.to_numpy()[same]) / lib.S_X2.to_numpy()[same])) if same.any() else None
    rows.append(dict(model=model, estimation=est, items_compared=int(same.sum()), items_collapsed_by_library=int((~same).sum()),
                     max_relative_diff=rel, passed=bool(same.any() and rel <= 1e-4)))
    print(rows[-1])
json.dump({"method": "independent Orlando-Thissen S-X2 (Lord-Wingersky, 61 GH nodes), compared on items without library cell collapsing", "cases": rows, "all_pass": all(r["passed"] for r in rows)}, open(sys.argv[1], "w"), indent=1)
