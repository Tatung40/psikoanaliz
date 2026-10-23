"""Cached, bounded GRM DIF study. Exports fitted summaries, never response rows."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import grm_dif as G
from scripts.grm_dif_validation import generate
from scripts.dif_fixed_anchor_simulation import binomial, write_json

ANCHORS = ["M1", "M2", "M3", "M4"]


def finite(value):
    return float(value) if np.isfinite(value) else None


def one_repetition(condition, replicate, seed, fit_fn=G.test_dif):
    generated_seed = int(np.random.SeedSequence([seed, int(condition == "both"), replicate]).generate_state(1)[0])
    x, groups, truth = generate(generated_seed, condition)
    record = dict(condition=condition, replicate=replicate, seed=generated_seed, status="inconclusive", items=[])
    started = perf_counter()
    try:
        out = fit_fn(x, groups, "R", "F", ANCHORS, points=61)
        t = out["table"].set_index("Madde")
        for i in (4, 5):
            name = x.columns[i]; row = t.loc[name]
            params = out["parameters"][out["parameters"].Madde == name]
            estimates = []
            for c, (_, p) in enumerate(params.iterrows()):
                known = truth["a"][1, i] - truth["a"][0, i] if c == 0 else truth["b"][1, i, c - 1] - truth["b"][0, i, c - 1]
                estimates.append(dict(parameter=p["Parametre"], truth=float(known), estimate=finite(p["Odak − referans"]),
                                      lower=finite(p["%95 GA alt"]), upper=finite(p["%95 GA üst"])))
            def integrand(theta):
                delta = np.sum(expit(truth["a"][1, i] * (theta - truth["b"][1, i]))
                               - expit(truth["a"][0, i] * (theta - truth["b"][0, i])))
                return delta * norm.pdf(theta, -.4, 1.2)
            effect_truth = quad(integrand, -np.inf, np.inf, epsabs=1e-10)[0]
            record["items"].append(dict(item=name, p=finite(row["p"]), q=finite(row["q (BH)"]),
                threshold_q=finite(row["q (BH, eşikler)"]), slope_q=finite(row["q (BH, eğim | eşikler)"]),
                parameters=estimates, effect=dict(truth=effect_truth, estimate=finite(row["İşaretli etki (puan)"]),
                    lower=finite(row["Etki %95 GA alt"]), upper=finite(row["Etki %95 GA üst"]))))
        full = out["fits"][4]["H2"]
        _, _, mu, sd = full["design"].unpack(full["params"])
        record.update(mu=float(mu), sd=float(sd), optimizations=out["n_fits"])
        if all(all(item[k] is not None for k in ("p", "q", "threshold_q", "slope_q")) for item in record["items"]):
            record["status"] = "valid"
        else:
            record["reasons"] = t.loc[["M5", "M6"], "Not"].tolist()
    except (ValueError, np.linalg.LinAlgError) as exc:
        record["reason"] = str(exc)
    record["seconds"] = perf_counter() - started
    return record


def recovery(values):
    values = [v for v in values if all(v[k] is not None for k in ("truth", "estimate", "lower", "upper"))]
    errors = [v["estimate"] - v["truth"] for v in values]
    return dict(n=len(values), bias=float(np.mean(errors)) if errors else None,
        rmse=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        pointwise_coverage=binomial(sum(v["lower"] <= v["truth"] <= v["upper"] for v in values), len(values)))


def summarize(records):
    output = {}
    for condition in ("null", "both"):
        all_rows = [r for r in records if r["condition"] == condition]
        valid = [r for r in all_rows if r["status"] == "valid"]
        result = dict(total=len(all_rows), valid=len(valid), inconclusive=len(all_rows) - len(valid),
            any_bh=binomial(sum(any(i["q"] < .05 for i in r["items"]) for r in valid), len(valid)), items={})
        for name in ("M5", "M6"):
            rows = [next(i for i in r["items"] if i["item"] == name) for r in valid]
            result["items"][name] = dict(
                raw_rejection=binomial(sum(r["p"] < .05 for r in rows), len(rows)),
                bh_rejection=binomial(sum(r["q"] < .05 for r in rows), len(rows)),
                threshold_bh=binomial(sum(r["threshold_q"] < .05 for r in rows), len(rows)),
                slope_bh=binomial(sum(r["slope_q"] < .05 for r in rows), len(rows)),
                parameters=[dict(parameter=label, **recovery([r["parameters"][c] for r in rows]))
                            for c, label in enumerate(("a", "b1", "b2"))],
                signed_effect=recovery([r["effect"] for r in rows]))
        result["latent_recovery"] = {name: dict(
            bias=float(np.mean([r[name] - truth for r in valid])) if valid else None,
            rmse=float(np.sqrt(np.mean([(r[name] - truth) ** 2 for r in valid]))) if valid else None)
            for name, truth in (("mu", -.4), ("sd", 1.2))}
        output[condition] = result
    return output


def run(cache_dir, repetitions=100, seed=20261024, fit_fn=G.test_dif):
    if repetitions < 1:
        raise ValueError("Positive repetition count required")
    paths = [Path(__file__), ROOT / "scripts/grm_dif_validation.py", ROOT / "core/grm_dif.py",
             ROOT / "core/rasch.py", ROOT / "core/irt_dif.py", ROOT / "core/rasch_dif.py"]
    config = dict(model="GRM", seed=seed, n_reference=600, n_focal=400, categories=3, points=61,
        audit_points=81, anchors=ANCHORS, latent=dict(reference=[0., 1.], focal=[-.4, 1.2]),
        effects=dict(M5="all thresholds +.6", M6="slope -.8"),
        python=sys.version.split()[0], versions={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "pandas")},
        source_sha256=hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest())
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    folder = Path(cache_dir) / key; folder.mkdir(parents=True, exist_ok=True)
    records, computed = [], 0
    for condition in ("null", "both"):
        for i in range(repetitions):
            path = folder / f"{condition}_{i:04d}.json"
            if path.exists():
                record = json.loads(path.read_text())
            else:
                record = one_repetition(condition, i, seed, fit_fn)
                write_json(path, record); computed += 1
            records.append(record)
    result = dict(config=config, repetitions_per_condition=repetitions, results=summarize(records),
        interpretation="Rates and pointwise coverage condition on valid fits; inconclusive fits are counted separately. Exact binomial Monte Carlo intervals. Complete-null any-BH estimates FDR for this two-item family, not general FDR control. Independent reference: Python mirt EM and analytic/adaptive-integral checks, NOT R mirt.")
    write_json(folder / "summary.json", result)
    return result, folder, computed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    _, folder, count = run(args.cache_dir, args.repetitions, args.seed)
    print(json.dumps(dict(summary=str(folder / "summary.json"), new_repetitions=count)))
