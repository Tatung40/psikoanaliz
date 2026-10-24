"""Independent-generator pilot for the ACTIVE library adapter, not certification.

Existing native DIF studies are not rerun. Two seeds measure single-group
library recovery separately from the already recorded concurrent DIF evidence.
"""
import json
from pathlib import Path
import sys
import importlib.metadata
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.irt_python_mirt_adapter import fit_python_mirt
from scripts.polytomous_dif_validation import generate
from scripts.grm_dif_validation import generate as generate_grm


def run():
    records = []
    for model in ("GRM", "PCM", "GPCM", "NRM"):
        for seed in (20261024, 20261025):
            record = dict(model=model, seed=seed, n_persons=600, n_items=6, categories=3, status="FAILED")
            try:
                if model == "GRM":
                    x, _, truth = generate_grm(seed=seed, condition="null")
                    known = dict(discrimination=truth["a"][0], thresholds=truth["b"][0])
                else:
                    x, _, truth = generate(model, seed, "null")
                    if model == "NRM": known = dict(slopes=truth["A"][0], intercepts=truth["D"][0])
                    else:
                        a = truth["A"][0, :, -1] / 2
                        known = dict(discrimination=a, steps=-np.diff(truth["D"][0], axis=1) / a[:, None])
                fit = fit_python_mirt(x.iloc[:600], model=model, n_categories=3).fit
                estimates = fit.model.parameters
                record["converged"] = bool(fit.converged)
                record["metrics"] = {}
                for name, actual in known.items():
                    estimated = np.asarray(estimates[name], float)
                    if model == "NRM":
                        estimated = estimated - estimated[:, :1]
                        if name == "slopes" and estimated[0, 1] < 0: estimated = -estimated
                    if estimated.shape != actual.shape or not np.isfinite(estimated).all(): raise ValueError("Invalid parameter schema")
                    residual = estimated - actual
                    record["metrics"][name] = dict(bias=float(residual.mean()), mae=float(abs(residual).mean()),
                        rmse=float(np.sqrt(np.mean(residual**2))), max_absolute_error=float(abs(residual).max()))
                record["status"] = "MEASURED" if record["converged"] else "NONCONVERGED"
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
    return dict(engine="Python mirt", version=importlib.metadata.version("mirt"),
        scope="Two-seed 3-category N600/6-item independent-generator single-group pilot; not general recovery/coverage certification",
        runs=records, RSM="UNAVAILABLE in the public fit_mirt adapter; existing native fixed-anchor RSM DIF is a different limited engine")


if __name__ == "__main__": print(json.dumps(run(), indent=2, allow_nan=False))
