import glob, json, sys
def load(d, pat): return {r["r"]: r for f in glob.glob(f"{d}/{pat}") for r in map(json.loads, open(f))}
old, new = load(sys.argv[1], "k_*.jsonl"), load(sys.argv[2], "k_*.jsonl")
keys = sorted(set(old) & set(new)); diff = []; n = 0
for k in keys:
    a, b = old[k].get("sets"), new[k].get("sets")
    if a is None or b is None:
        if (a is None) != (b is None): diff.append({"r": k, "refusal_mismatch": True})
        continue
    for s, t in zip(a, b):
        n += 1
        if s != t: diff.append({"r": k, "item": s["item"], "param": s["param"], "frozen": s, "optimised": t})
rep = {"datasets_compared": len(keys), "missing": sorted(set(old) ^ set(new)), "sets_compared": n, "sets_different": len(diff), "differences": diff[:50]}
json.dump(rep, open(sys.argv[3], "w"), indent=1); print("COMPARE " + json.dumps(rep))
