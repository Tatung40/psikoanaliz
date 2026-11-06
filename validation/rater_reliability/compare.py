"""Compare production results with the R reference CSV (PROTOCOL.json tolerances) and write evidence.json.
PYTHONPATH=. python -m validation.rater_reliability.compare <reference.csv> [--write]
"""
import argparse, json, math
from pathlib import Path
import pandas as pd

import core.rater_reliability as R

ROOT = Path(__file__).resolve().parents[2]; EX = ROOT / "examples/rater_reliability"; HERE = Path(__file__).resolve().parent
KEY = {"estimate": ("estimate", lambda r: r["estimate"]), "se": ("se", lambda r: r["se"]), "z0": ("z0", lambda r: r["z0"]),
       "p": ("p", lambda r: r["p"]), "ci_low": ("ci95", lambda r: r["ci95"][0]), "ci_high": ("ci95", lambda r: r["ci95"][1]),
       "F": ("F", lambda r: r["F"]), "df1": ("df1", lambda r: r["df1"]), "df2": ("df2", lambda r: r["df2"]),
       "chi2": ("chi2", lambda r: r["chi2"]), "df": ("df", lambda r: r["df"]), "all_agree": ("all_agree", lambda r: r["all_agree"]),
       "bias": ("bias", lambda r: r["bias"]), "sd": ("sd", lambda r: r["sd"]), "loa_low": ("loa", lambda r: r["loa"][0]),
       "loa_high": ("loa", lambda r: r["loa"][1]), "bias_ci_low": ("ci95_bias", lambda r: r["ci95_bias"][0]),
       "bias_ci_high": ("ci95_bias", lambda r: r["ci95_bias"][1]), "loa_low_ci_low": ("ci95_loa", lambda r: r["ci95_loa"][0][0]),
       "loa_high_ci_high": ("ci95_loa", lambda r: r["ci95_loa"][1][1])}


def tolerance(reference, key, value, method):
    if reference.startswith("irrCAC"): return 5e-6 + 1e-9
    if key == "p": return 1e-9 + 1e-6 * abs(value)
    if key.startswith("ci_") and method.startswith("icc_"): return 1e-8
    if reference == "analytic": return 1e-10
    return 1e-9


def load(example):
    df = pd.read_csv(EX / example["file"]); c = example["config"]
    return R.from_counts(df, c["id_col"], c["category_cols"]) if c["layout"] == "counts" else R.from_wide(df, c["id_col"], c["rater_cols"])


def compare(ref_csv):
    ref = pd.read_csv(ref_csv); reg = json.loads((EX / "registry.json").read_text(encoding="utf-8"))
    rows, status = [], {}
    for e in reg["examples"]:
        res = R.compute(e["method"], load(e)); sub = ref[ref.example == e["id"]]
        for r in sub.itertuples():
            out, get = KEY[r.key]; mine = float(get(res)); tol = tolerance(r.reference, r.key, r.value, e["method"])
            diff = abs(mine - r.value) if math.isfinite(mine) and math.isfinite(r.value) else float("inf")
            rows.append(dict(example=e["id"], method=e["method"], reference=r.reference, key=r.key, output=out,
                             python=mine, r=r.value, abs_diff=diff, tolerance=tol, ok=bool(diff <= tol)))
    df = pd.DataFrame(rows)
    for m in R.METHODS:
        s = df[df.method == m]
        outs = {o: bool(g.ok.all()) for o, g in s.groupby("output")}
        primary = next((o for o in ("estimate", "chi2", "bias") if o in outs), None)
        label = ("UNSUPPORTED" if s.empty else "FAIL" if primary is None or not outs[primary] else
                 "NUMERICALLY VALIDATED" if all(outs.values()) else "NUMERICALLY VALIDATED (kısmi; başarısız çıktılar gizli)")
        status[m] = dict(label=label, validated_outputs=sorted(o for o, ok in outs.items() if ok),
                         failed_outputs=sorted(o for o, ok in outs.items() if not ok),
                         references=sorted(s.reference.unique()), max_abs_diff=float(s.abs_diff.max()) if len(s) else None)
    return df, status, reg


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("reference"); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    df, status, reg = compare(a.reference)
    print(df.groupby("method").agg(n=("ok", "size"), ok=("ok", "all"), max_diff=("abs_diff", "max")).to_string())
    print(df[~df.ok].to_string() if (~df.ok).any() else "all comparisons within tolerance")
    if a.write:
        df.to_csv(HERE / "comparison.csv", index=False)
        (HERE / "evidence.json").write_text(json.dumps(dict(protocol="validation/rater_reliability/PROTOCOL.json", status=status,
            n_comparisons=len(df), max_abs_diff=float(df.abs_diff.max()), scope="examples/rater_reliability fixtures only; not CERTIFIED"),
            ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        best = {}
        for r in df.itertuples():                              # expected values: first unrounded reference per key
            if not r.reference.startswith("irrCAC"):
                best.setdefault((r.example, r.key), (r.r, r.reference))
        for e in reg["examples"]:
            e["expected"] = {k: {"value": v, "source": s} for (ex, k), (v, s) in best.items() if ex == e["id"]} or \
                {r.key: {"value": r.r, "source": r.reference} for r in df[df.example == e["id"]].itertuples()}
        (EX / "registry.json").write_text(json.dumps(reg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
