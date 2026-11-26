"""Gate evaluation for closure K (exact denominators, dataset-clustered MCSE). No fits."""
import glob, json, sys
import numpy as np

def mcse_ratio(num, den):
    r = num.sum() / den.sum(); z = (num - r * den) / den.mean(); return float(r), float(z.std(ddof=1) / np.sqrt(len(num)))

rows = [json.loads(l) for f in sorted(glob.glob(f"{sys.argv[1]}/k_*.jsonl")) for l in open(f)]
R = len(rows); rep = {"R": R, "failed_datasets": sum("sets" not in r for r in rows)}
for q in "cd":
    cov = np.zeros((R, 12), bool); exact = np.zeros((R, 12), bool); whole = np.zeros((R, 12), bool); empty = np.zeros((R, 12), bool); width = np.full((R, 12), np.nan)
    for i, r in enumerate(rows):
        for s in r.get("sets", []):
            if s["param"] != q: continue
            j = s["item"]; cov[i, j], exact[i, j], whole[i, j], empty[i, j], width[i, j] = s["covered_set"], s["covered_exact"], s["whole"], s["empty"], s["width"]
    ok = np.array(["sets" in r for r in rows]); info = ok[:, None] & ~whole & ~empty
    u, ue = mcse_ratio(cov.sum(1).astype(float), np.full(R, 12.0)); c, ce = mcse_ratio((cov & info).sum(1).astype(float), info.sum(1).astype(float))
    fail = 1 - ok.mean(); wf = whole.sum() / (12 * R); mw = float(np.nanmedian(width[info]))
    gates = {k: bool(v) for k, v in {"unconditional[.93,.97]": .93 <= u <= .97, "MCSE<=.004": ue <= .004, "conditional[.93,.97]": .93 <= c <= .97,
             "conditional_MCSE<=.004": ce <= .004, "whole_box<=.5": wf <= .5, "median_informative_width<.5": mw < .5, "numerical_failure<=.01": fail <= .01}.items()}
    rep[q] = dict(item_sets=12 * R, covered=int(cov.sum()), informative=int(info.sum()), covered_informative=int((cov & info).sum()),
                  whole_box=int(whole.sum()), empty=int(empty.sum()), unconditional=u, unconditional_MCSE=ue, conditional=c, conditional_MCSE=ce,
                  exact_T_coverage=float(exact.mean()), whole_box_fraction=float(wf), median_informative_width=mw,
                  per_item_coverage=cov.mean(0).round(4).tolist(), per_item_whole=whole.mean(0).round(4).tolist(), per_item_empty=empty.mean(0).round(4).tolist(),
                  gates=gates, CERTIFIED=all(gates.values()))
json.dump(rep, open(sys.argv[2], "w"), indent=1); print("REPORT_K " + json.dumps(rep))
