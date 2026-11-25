"""Read-only aggregation of confirmation JSONL against PROTOCOL.json gates (dataset = Monte Carlo cluster). No fits."""
import glob, json, sys
import numpy as np
from scipy import stats

Z = stats.norm.ppf(.975)


def cp_upper(k, n, a): return 1.0 if k == n else float(stats.beta.ppf(1 - a, k + 1, n - k))


def cp_lower(k, n, a): return 0.0 if k == 0 else float(stats.beta.ppf(a, k, n - k + 1))


def load(pattern):
    return [json.loads(l) for f in sorted(glob.glob(pattern)) for l in open(f)]


def p3(null, alt):
    out = {}
    for m in ("3PL", "4PL"):
        R = [r for r in null if r["model"] == m]; n = len(R)
        conv = [r for r in R if r.get("converged") and "p_A" in r]
        pa = np.full((n, 12), np.nan); pc = np.full((n, 12), np.nan); ident = 0.0; pooled_rows = np.zeros(12)
        for i, r in enumerate(R):
            if r in conv:
                pa[i] = np.array(r["p_A"], float); pc[i] = np.array(r["p_chi"], float)
                pa[i][~np.isfinite(pc[i])] = np.nan                     # production non-evaluable stays non-evaluable
                ident = max(ident, float(np.max(np.abs(np.array(r["sx"]) - np.array(r["sx_native"])))))
                pooled_rows += np.array([len(g) < 11 or any(len(x) > 1 for x in g) for g in r["groups"]])
        noneval = ~np.isfinite(pa); rej = (pa < .05) | noneval                 # worst case: non-evaluable = rejection
        per_item = rej.mean(0); a = .05 / 24
        upper = [cp_upper(int(k), n, a) for k in rej.sum(0)]
        pooled = rej.mean(); cl = rej.mean(1); mcse = float(cl.std(ddof=1) / np.sqrt(n))
        A = [r for r in alt if r["model"] == m]; na = len(A)
        ra = np.array([(r.get("p_A") or [np.nan])[0] for r in A], float); rc = np.array([(r.get("p_chi") or [np.nan])[0] for r in A], float)
        powA = int(np.sum(ra < .05)); pw_low = cp_lower(powA, na, .025)
        gates = {"per_item_upper<=.07": max(upper) <= .07, "pooled_MCSE<=.004": mcse <= .004,
                 "pooled_lower>=.03": pooled - Z * mcse >= .03, "non_evaluable<=.01": float(noneval.mean()) <= .01,
                 "identity<=1e-8": ident <= 1e-8, "power_lower>.10": pw_low > .10, "converged_all": len(conv) == n}
        out[m] = dict(R=n, converged=len(conv), pooled_typeI=pooled, pooled_MCSE=mcse, per_item=per_item.tolist(),
                      per_item_simultaneous_upper=upper, non_evaluable=float(noneval.mean()), identity_max=ident,
                      original_chi_pooled=float(np.nanmean(pc < .05)), pooling_fraction_per_item=(pooled_rows / max(len(conv), 1)).tolist(),
                      misfit_R=na, power_A=powA / max(na, 1), power_A_lower=pw_low, power_chi=float(np.nanmean(rc < .05)),
                      gates=gates, passed=all(gates.values()))
    return out


def p1(null):
    out = {}; box = {"c": (0, .5), "d": (.5, 1)}
    for m in ("3PL", "4PL"):
        k = 4 if m == "4PL" else 3; R = [r for r in null if r["model"] == m]; n = len(R)
        t = np.array(next(r["truth"] for r in R if "truth" in r)).reshape(12, k)
        hit = np.zeros((n, 12, k)); ref = np.zeros(n, bool); est = np.full((n, 12, k), np.nan); se = np.full((n, 12, k), np.nan)
        lo = np.full((n, 12, k), np.nan); hi = np.full((n, 12, k), np.nan)
        for i, r in enumerate(R):
            if "one_step" not in r or not r.get("pd_I") or r.get("cond_I", np.inf) > 1e12: ref[i] = True; continue
            e = np.array(r["one_step"]).reshape(12, k); s = np.array(r["se_ml"]).reshape(12, k)
            l, h = e - Z * s, e + Z * s
            l[:, 0] = np.maximum(l[:, 0], 0)
            for q, (a, b) in box.items():
                if "abcd".index(q) < k: j = "abcd".index(q); l[:, j] = np.clip(l[:, j], a, b); h[:, j] = np.clip(h[:, j], a, b)
            hit[i] = (l <= t) & (t <= h); est[i], se[i], lo[i], hi[i] = e, s, l, h
        res = {}
        for j, q in enumerate("abcd"[:k]):
            cov_c = hit[:, :, j].mean(1); cov = cov_c.mean(); mcse = cov_c.std(ddof=1) / np.sqrt(n)
            w = (hi - lo)[:, :, j][~ref]
            ok = ~ref; sd = np.nanstd(est[ok][:, :, j], 0, ddof=1); ms = np.nanmean(se[ok][:, :, j], 0)
            # cluster delta CI for pooled ratio mean(SE)/SD: bootstrap-free delta via per-dataset contributions
            E, Sx = est[ok][:, :, j], se[ok][:, :, j]; me, v = E.mean(0), E.var(0, ddof=1); rj = Sx.mean(0) / np.sqrt(v)
            inf = (rj * ((Sx - Sx.mean(0)) / Sx.mean(0) - ((E - me)**2 - v) / (2 * v))).mean(1)   # delta IF per dataset
            ratio = float(rj.mean()); rse = float(inf.std(ddof=1) / np.sqrt(len(inf)))
            g = {"coverage": abs(cov - .95) <= .02, "MCSE<=.004": mcse <= .004, "refusal<=.01": ref.mean() <= .01}
            info = {"median_width": float(np.median(w))}
            if q in box:
                L = box[q][1] - box[q][0]; full = float(np.mean(w >= L - 1e-12))
                info.update(full_box_fraction=full); g["informative"] = bool(np.median(w) < L and full <= .5)
            res[q] = dict(coverage=float(cov), MCSE=float(mcse), per_item=hit[:, :, j].mean(0).tolist(),
                          bias_onestep=float(np.nanmean(est[ok][:, :, j] - t[:, j])), mean_SE=float(ms.mean()), emp_SD=float(sd.mean()),
                          SE_over_SD=ratio, SE_over_SD_CI=[ratio - Z * rse, ratio + Z * rse],
                          SE_certified=bool(ratio - Z * rse >= .9245 and ratio + Z * rse <= 1.107), **info, gates=g, CI_passed=all(g.values()))
        out[m] = dict(R=n, refusal=float(ref.mean()), params=res)
    return out


def p3_boot(null, alt):
    out = {}
    for m in ("3PL", "4PL"):
        R = [r for r in null if r["model"] == m]; n = len(R)
        if not n: out[m] = {"executed": False}; continue
        pb = np.array([r.get("p_boot") or [np.nan] * 12 for r in R], float)
        noneval = ~np.isfinite(pb); rej = (pb <= .05) | noneval
        upper = [cp_upper(int(k), n, .05 / 24) for k in rej.sum(0)]; cl = rej.mean(1); mcse = float(cl.std(ddof=1) / np.sqrt(n))
        A = [r for r in alt if r["model"] == m]; ra = np.array([(r.get("p_boot") or [np.nan])[0] for r in A], float)
        k = int(np.sum((ra <= .05) | ~np.isfinite(ra))); low = cp_lower(int(np.sum(ra <= .05)), len(A), .025)
        gates = {"per_item_upper<=.07": max(upper) <= .07, "pooled_MCSE<=.004": mcse <= .004, "pooled_lower>=.03": bool(rej.mean() - Z * mcse >= .03),
                 "non_evaluable<=.01": float(noneval.mean()) <= .01, "power_lower>.10": low > .10}
        out[m] = dict(executed=True, R=n, pooled_typeI=float(rej.mean()), pooled_MCSE=mcse, per_item=rej.mean(0).tolist(), per_item_simultaneous_upper=upper,
                      non_evaluable=float(noneval.mean()), inner_failures=int(sum(r.get("inner_failures", 0) for r in R)),
                      misfit_R=len(A), power=float(np.mean(ra <= .05)), power_lower=low, gates=gates, passed=all(gates.values()))
    return out


if __name__ == "__main__":
    d = sys.argv[1]
    null, alt = load(f"{d}/null_*.jsonl"), load(f"{d}/alt_*.jsonl")
    rep = {"P3_method_A": p3(null, alt), "P1_onestep": p1(null), "P3_bootstrap": p3_boot(load(f"{d}/bnull_*.jsonl"), load(f"{d}/balt_*.jsonl"))}
    json.dump(rep, open(sys.argv[2], "w"), indent=1, default=float)
    print(json.dumps({m: {"P3": rep["P3_method_A"][m]["passed"], **{q: v["CI_passed"] for q, v in rep["P1_onestep"][m]["params"].items()}} for m in ("3PL", "4PL")}))
