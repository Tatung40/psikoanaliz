"""Resumable fixed-anchor studies; exact MC intervals, failed fits explicit."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import softmax
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import polytomous_dif as P
from scripts.polytomous_dif_validation import generate, ANCHORS
from scripts.dif_fixed_anchor_simulation import binomial, write_json
from scripts.grm_dif_study import recovery, finite


def known_effects(model, truth):
    effects = {}
    for i in (4, 5):
        def integrand(theta):
            pr = softmax(truth["A"][0, i] * theta + truth["D"][0, i])
            pf = softmax(truth["A"][1, i] * theta + truth["D"][1, i])
            delta = pf - pr
            return (delta if model == "NRM" else np.array([delta @ np.arange(3)])) * norm.pdf(theta, -.4, 1.2)
        effects[i] = quad_vec(integrand, -np.inf, np.inf, epsabs=1e-10)[0]
    return effects


def one_repetition(model, condition, replicate, seed, known, fit_fn=P.test_dif):
    generated_seed = int(np.random.SeedSequence([seed, P.MODELS.index(model), int(condition == "both"), replicate]).generate_state(1)[0])
    x, groups, truth = generate(model, generated_seed, condition)
    record = dict(model=model, condition=condition, replicate=replicate, seed=generated_seed, status="inconclusive", items=[])
    started = perf_counter()
    try:
        out = fit_fn(x, groups, "R", "F", ANCHORS, model=model, points=61)
        table = out["table"].set_index("Madde")
        for i in (4, 5):
            name = x.columns[i]; row = table.loc[name]
            d = out["fits"][i]["H2"]["design"]
            true_fit = dict(design=d, params=d.encode(truth["A"], truth["D"], -.4, 1.2), cov=None)
            true_parameters = P.parameter_contrasts(true_fit, i, False)["Odak − referans"].to_numpy()
            parameters = out["parameters"].loc[lambda f: f.Madde == name]
            effects = out["effects"].loc[lambda f: f.Madde == name]
            def estimates(frame, truths, label):
                return [dict(parameter=r[label], truth=float(t), estimate=finite(r["Odak − referans"]),
                             lower=finite(r["%95 GA alt"]), upper=finite(r["%95 GA üst"]))
                        for (_, r), t in zip(frame.iterrows(), truths)]
            record["items"].append(dict(item=name, p=finite(row.p),
                families={c: finite(row[c]) for c in table if c.startswith("q (BH")},
                parameters=estimates(parameters, true_parameters, "Parametre"),
                effects=estimates(effects, known[i], "Etki")))
        full = out["fits"][4]["H2"]
        _, _, mu, sd = full["design"].unpack(full["params"])
        record.update(mu=mu, sd=sd, optimizations=out["n_fits"])
        if all(all(q is not None for q in item["families"].values()) for item in record["items"]): record["status"] = "valid"
        else: record["reasons"] = table.loc[["M5", "M6"], "Not"].tolist()
    except (ValueError, np.linalg.LinAlgError) as exc: record["reason"] = str(exc)
    record["seconds"] = perf_counter() - started
    return record


def summarize(records):
    output = {}
    for condition in ("null", "both"):
        rows = [r for r in records if r["condition"] == condition]
        valid = [r for r in rows if r["status"] == "valid"]
        result = dict(total=len(rows), valid=len(valid), inconclusive=len(rows) - len(valid),
                      inconclusive_fraction=binomial(len(rows) - len(valid), len(rows)), items={})
        families = valid[0]["items"][0]["families"] if valid else []
        result["any_bh"] = {f: binomial(sum(any(i["families"][f] < .05 for i in r["items"]) for r in valid), len(valid)) for f in families}
        for name in ("M5", "M6"):
            items = [next(i for i in r["items"] if i["item"] == name) for r in valid]
            entry = dict(raw_rejection=binomial(sum(i["p"] < .05 for i in items), len(items)),
                bh_rejection={f: binomial(sum(i["families"][f] < .05 for i in items), len(items)) for f in families})
            for key in ("parameters", "effects"):
                entry[key] = [dict(parameter=r["parameter"], **recovery([i[key][j] for i in items])) for j, r in enumerate(items[0][key])] if items else []
            result["items"][name] = entry
        result["latent_recovery"] = {key: dict(bias=float(np.mean([r[key] - truth for r in valid])) if valid else None,
            rmse=float(np.sqrt(np.mean([(r[key] - truth) ** 2 for r in valid]))) if valid else None)
            for key, truth in (("mu", -.4), ("sd", 1.2))}
        output[condition] = result
    return output


def run(model, cache_dir, repetitions=100, seed=20261025, fit_fn=P.test_dif):
    if model not in P.MODELS or repetitions < 1: raise ValueError("Invalid model/repetitions")
    paths = [Path(__file__), ROOT / "scripts/polytomous_dif_validation.py", ROOT / "core/polytomous_dif.py",
             ROOT / "core/grm_dif.py", ROOT / "core/rasch.py", ROOT / "core/irt_dif.py", ROOT / "core/rasch_dif.py"]
    config = dict(model=model, seed=seed, n_reference=600, n_focal=400, categories=3, points=61,
        audit_points=81, anchors=ANCHORS, latent=dict(reference=[0., 1.], focal=[-.4, 1.2]),
        python=sys.version.split()[0], versions={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "pandas")},
        source_sha256=hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest())
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    folder = Path(cache_dir) / key; folder.mkdir(parents=True, exist_ok=True)
    records, computed = [], 0
    for condition in ("null", "both"):
        truth = generate(model, seed, condition)[2]; known = known_effects(model, truth)
        for i in range(repetitions):
            path = folder / f"{condition}_{i:04d}.json"
            if path.exists(): record = json.loads(path.read_text())
            else:
                record = one_repetition(model, condition, i, seed, known, fit_fn)
                write_json(path, record); computed += 1
            records.append(record)
    summary = dict(config=config, cache_key=key, computed=computed, reused=len(records) - computed,
        optimizer_fits=sum(r.get("optimizations", 0) for r in records),
        fit_seconds=sum(r["seconds"] for r in records), conditions=summarize(records),
        limitation="EXPERIMENTAL. Rates/coverage condition on valid fits; inconclusives explicit. Narrow 3-category/6-item/N600+400/4-true-anchor design. General FDR, selected anchors and broad coverage NOT established. No R execution.")
    write_json(folder / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=P.MODELS)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--cache", type=Path, default=ROOT.parent / ".cache/psikoanaliz/polytomous_dif")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.model, args.cache, args.repetitions)
    write_json(args.output, summary)
    print(json.dumps({k: summary[k] for k in ("cache_key", "computed", "reused", "optimizer_fits", "fit_seconds")}))
    print(json.dumps({k: {j: v for j, v in r.items() if j in ("valid", "inconclusive", "any_bh")} for k, r in summary["conditions"].items()}))
