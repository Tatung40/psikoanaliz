import glob, json, sys
import numpy as np
P = json.load(open("validation/p4_confirm/PROTOCOL.json")); R = P["R"]["value"]
rows = [json.loads(l) for f in glob.glob(f"{sys.argv[1]}/*.jsonl") for l in open(f)]
BLOCKS = {"GRM": ("discrimination", "thresholds"), "PCM": ("steps",), "GPCM": ("discrimination", "steps"), "NRM": ("slopes", "intercepts")}
rep = {}
for m, blocks in BLOCKS.items():
    rs = [r for r in rows if r["model"] == m]; npar = {}
    for r in rs:
        for b in blocks:
            if r.get("blocks", {}).get(b): npar[b] = r["blocks"][b]["n_param"]
    for b in blocks:
        num = np.array([r["blocks"][b]["covered"] if r.get("status") == "MEASURED" and r.get("converged") and r.get("blocks", {}).get(b) else 0 for r in rs], float)
        nci = np.array([r["blocks"][b]["n_ci"] if r.get("status") == "MEASURED" and r.get("blocks", {}).get(b) else 0 for r in rs], float)
        den = np.full(len(rs), float(npar[b])); c = num.sum() / den.sum(); z = (num - c * den) / den.mean(); mc = float(z.std(ddof=1) / np.sqrt(len(rs)))
        refusal = 1 - nci.sum() / den.sum()
        g = {"R": len(rs) == R, "coverage": .925 <= c <= .975, "simultaneous": .925 <= c - 2.690 * mc and c + 2.690 * mc <= .975, "mcse<=.004": mc <= .004, "refusal<=.01": refusal <= .01}
        rep[f"{m} {b}"] = dict(R=len(rs), coverage=c, cluster_mcse=mc, refusal=refusal, nonconverged=sum(not r.get("converged") for r in rs),
                              failed=sum(r.get("status") != "MEASURED" for r in rs), gates=g, CERTIFIED=all(g.values()))
json.dump(rep, open(sys.argv[2], "w"), indent=1); print("P4C1 " + json.dumps(rep))
