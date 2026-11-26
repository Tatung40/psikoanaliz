"""Read-only audit of the archived P1 closure-J confirmation (run 45665227625): exact denominators, whole-box
intervals, unconditional and conditional (non-whole-box) coverage with dataset-clustered MCSE. No fits."""
import glob, json, sys
import numpy as np

Z = 1.959964; BOX = {"c": (0.0, .5), "d": (.5, 1.0)}


def ratio_mcse(num, den):
    """Dataset-clustered delta-method SE of sum(num)/sum(den)."""
    n = len(num); r = num.sum() / den.sum(); z = (num - r * den) / den.mean()
    return float(r), float(z.std(ddof=1) / np.sqrt(n))


def audit(rows, model):
    k = 4 if model == "4PL" else 3; R = [r for r in rows if r["model"] == model]; n = len(R)
    t = np.array(next(r["truth"] for r in R if "truth" in r)).reshape(12, k)
    out = {"datasets": n}
    ok = np.array(["one_step" in r and r.get("pd_I") and r.get("cond_I", np.inf) <= 1e12 for r in R]); out["refused_datasets"] = int((~ok).sum())
    E = np.full((n, 12, k), np.nan); S = np.full((n, 12, k), np.nan)
    for i, r in enumerate(R):
        if ok[i]: E[i] = np.reshape(r["one_step"], (12, k)); S[i] = np.reshape(r["se_ml"], (12, k))
    for j, q in enumerate("abcd"[:k]):
        lo, hi = E[:, :, j] - Z * S[:, :, j], E[:, :, j] + Z * S[:, :, j]
        if q == "a": lo = np.maximum(lo, 0)
        if q in BOX: lo, hi = np.clip(lo, *BOX[q]), np.clip(hi, *BOX[q])
        cov = (lo <= t[:, j]) & (t[:, j] <= hi) & ok[:, None]
        full = np.zeros_like(cov) if q not in BOX else ((lo <= BOX[q][0] + 1e-12) & (hi >= BOX[q][1] - 1e-12) & ok[:, None])
        info = ok[:, None] & ~full
        u, ue = ratio_mcse(cov.sum(1).astype(float), np.full(n, 12.0))
        c, ce = ratio_mcse((cov & info).sum(1).astype(float), info.sum(1).astype(float)) if info.sum() else (float("nan"), float("nan"))
        w = (hi - lo)[ok]
        out[q] = dict(item_intervals=int(12 * n), refused_item_intervals=int(12 * (~ok).sum()), whole_box=int(full.sum()),
                      informative_intervals=int(info.sum()), covered=int(cov.sum()), covered_informative=int((cov & info).sum()),
                      unconditional_coverage=u, unconditional_MCSE=ue, conditional_coverage_non_whole_box=c, conditional_MCSE=ce,
                      whole_box_fraction=float(full.sum() / (12 * n)), median_width=float(np.median(w)),
                      median_width_informative=float(np.median((hi - lo)[info])) if info.sum() else None,
                      per_item_whole_box=full.mean(0).round(4).tolist(), per_item_conditional=[float((cov & info)[:, i].sum() / max(info[:, i].sum(), 1)) for i in range(12)])
    return out


if __name__ == "__main__":
    rows = [json.loads(l) for f in sorted(glob.glob(f"{sys.argv[1]}/null_*.jsonl")) for l in open(f)]
    rep = {m: audit(rows, m) for m in ("3PL", "4PL")}
    json.dump(rep, open(sys.argv[2], "w"), indent=1); print("AUDIT_JSON " + json.dumps(rep))
