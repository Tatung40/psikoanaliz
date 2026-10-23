"""Resumable, bounded 2PL fixed-anchor study; no raw responses exported.

Run with one BLAS thread. Default: 100 independent repetitions per condition;
exact binomial intervals expose the resulting Monte Carlo uncertainty. This is
evidence for this design, not general certification of Type I error or power.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.irt_dif import dif_analysis

ITEMS = [f"M{i + 1}" for i in range(12)]
TARGETS = ("M3", "M7")
ANCHORS = [name for name in ITEMS if name not in TARGETS]


def generate(seed, condition, replicate, n_reference=800, n_focal=500):
    if condition not in ("null", "dif"):
        raise ValueError("condition must be null or dif")
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(condition == "dif"), replicate]))
    a = np.linspace(.8, 1.8, 12)
    b = np.linspace(-1.8, 1.8, 12)
    focal_a, focal_b = a.copy(), b.copy()
    if condition == "dif":
        focal_b[2] += .6
        focal_a[6] -= .9
    parts = []
    for n, mu, sd, aa, bb in ((n_reference, 0., 1., a, b),
                               (n_focal, -.4, 1.2, focal_a, focal_b)):
        theta = rng.normal(mu, sd, n)
        p = expit(aa * (theta[:, None] - bb))
        parts.append((rng.random(p.shape) < p).astype(int))
    truth = {ITEMS[i]: {"a": float(focal_a[i] - a[i]), "b": float(focal_b[i] - b[i])}
             for i in (2, 6)}
    return (pd.DataFrame(np.vstack(parts), columns=ITEMS),
            ["R"] * n_reference + ["F"] * n_focal, truth)


def number(value):
    return float(value) if np.isfinite(value) else None


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def replicate_result(seed, condition, replicate, n_reference, n_focal, fit_fn=dif_analysis):
    x, groups, truth = generate(seed, condition, replicate, n_reference, n_focal)
    record = {"condition": condition, "replicate": replicate, "items": [], "status": "inconclusive"}
    started = perf_counter()
    try:
        fitted = fit_fn(x, groups, "R", "F", model="2PL", purify=False, fixed_anchors=ANCHORS)
        table = fitted["table"].set_index("Madde")
        for item in TARGETS:
            row = table.loc[item]
            entry = {"item": item, "p": number(row["p"]), "q": number(row["q (BH)"]),
                     "type": row.get("DIF türü", "")}
            for parameter in ("a", "b"):
                entry[parameter] = {
                    "estimate": number(row[f"Δ{parameter} ({parameter}_odak − {parameter}_ref)"]),
                    "truth": truth[item][parameter],
                    "lower": number(row[f"Δ{parameter} %95 GA alt"]),
                    "upper": number(row[f"Δ{parameter} %95 GA üst"])}
            record["items"].append(entry)
        joint = fitted["joint"]
        record.update(mu_f=number(joint["mu_F"]), sd_f=number(joint["sd_F"]), n_fits=fitted["n_fits"])
        if joint["converged"] and joint["cov"] is not None and all(r["p"] is not None for r in record["items"]):
            record["status"] = "valid"
    except (ValueError, np.linalg.LinAlgError) as exc:
        record["reason"] = str(exc)
    record["seconds"] = perf_counter() - started
    return record


def binomial(events, trials):
    if not trials:
        return {"events": events, "trials": trials, "estimate": None, "mc_ci95": [None, None]}
    result = binomtest(events, trials)
    ci = result.proportion_ci(confidence_level=.95, method="exact")
    return {"events": events, "trials": trials, "estimate": events / trials, "mc_ci95": [ci.low, ci.high]}


def summarize(records):
    summary = {}
    for condition in ("null", "dif"):
        all_records = [r for r in records if r["condition"] == condition]
        valid = [r for r in all_records if r["status"] == "valid"]
        item_summary = {}
        for item in TARGETS:
            rows = [next(e for e in r["items"] if e["item"] == item) for r in valid]
            effects = {}
            for parameter in ("a", "b"):
                usable = [r[parameter] for r in rows if all(r[parameter][k] is not None
                          for k in ("estimate", "lower", "upper"))]
                errors = [e["estimate"] - e["truth"] for e in usable]
                effects[parameter] = {"n": len(usable),
                    "bias": float(np.mean(errors)) if errors else None,
                    "rmse": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                    "pointwise_ci_coverage": binomial(sum(e["lower"] <= e["truth"] <= e["upper"]
                                                          for e in usable), len(usable))}
            item_summary[item] = {
                "raw_rejection": binomial(sum(r["p"] < .05 for r in rows), len(rows)),
                "bh_rejection": binomial(sum(r["q"] < .05 for r in rows), len(rows)),
                "parameter_contrasts": effects}
        summary[condition] = {"total_replicates": len(all_records), "valid_replicates": len(valid),
            "inconclusive_replicates": len(all_records) - len(valid),
            "any_bh_rejection": binomial(sum(any(e["q"] < .05 for e in r["items"]) for r in valid), len(valid)),
            "latent_recovery": {name: {
                "bias": float(np.mean([r[name] - truth for r in valid])) if valid else None,
                "rmse": float(np.sqrt(np.mean([(r[name] - truth) ** 2 for r in valid]))) if valid else None}
                for name, truth in (("mu_f", -.4), ("sd_f", 1.2))},
            "items": item_summary}
    return summary


def run(cache_dir, repetitions=100, seed=20261023, n_reference=800, n_focal=500, fit_fn=dif_analysis):
    if repetitions < 1 or min(n_reference, n_focal) < 50:
        raise ValueError("Positive repetitions and at least 50 persons per group required")
    sources = [Path(__file__), ROOT / "core/irt_dif.py", ROOT / "core/irt_mg.py",
               ROOT / "core/rasch.py", ROOT / "core/rasch_dif.py"]
    config = {"seed": seed, "n_reference": n_reference, "n_focal": n_focal,
        "model": "2PL", "quadrature": "gauss-hermite", "points": 41, "fixed_anchors": ANCHORS,
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "pandas")},
        "python": sys.version.split()[0],
        "source_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    folder = Path(cache_dir) / fingerprint
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    computed = 0
    for condition in ("null", "dif"):
        for i in range(repetitions):
            path = folder / f"{condition}_{i:04d}.json"
            if path.exists():
                record = json.loads(path.read_text())
            else:
                record = replicate_result(seed, condition, i, n_reference, n_focal, fit_fn)
                write_json(path, record)
                computed += 1
            records.append(record)
    output = {"config": config, "repetitions_per_condition": repetitions,
        "interpretation": "Rates and pointwise Wald coverage are conditional on valid fits in this design. Inconclusive replicates are reported separately. Exact binomial intervals quantify Monte Carlo uncertainty. Under the complete null, any BH rejection estimates FDR for the two studied items; not a general FDR guarantee.",
        "results": summarize(records)}
    write_json(folder / "summary.json", output)
    return output, folder, computed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--n-reference", type=int, default=800)
    parser.add_argument("--n-focal", type=int, default=500)
    args = parser.parse_args()
    _, folder, computed = run(args.cache_dir, args.repetitions, args.seed, args.n_reference, args.n_focal)
    print(json.dumps({"summary": str(folder / "summary.json"), "new_replicates": computed}))
