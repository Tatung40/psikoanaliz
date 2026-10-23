"""Deterministic GRM validation data; category generation is independent of the fitter."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def generate(seed=20261024, condition="both", n_reference=600, n_focal=400):
    if condition not in ("null", "threshold", "slope", "both"):
        raise ValueError("Unknown GRM condition")
    rng = np.random.default_rng(seed)
    names = [f"M{i + 1}" for i in range(6)]
    a = np.linspace(.9, 1.6, 6)
    b = np.linspace(-.6, .6, 6)[:, None] + np.array([-.8, .8])
    af, bf = a.copy(), b.copy()
    if condition in ("threshold", "both"):
        bf[4] += .6
    if condition in ("slope", "both"):
        af[5] -= .8
    parts = []
    for n, mu, sd, aa, bb in ((n_reference, 0., 1., a, b), (n_focal, -.4, 1.2, af, bf)):
        theta = rng.normal(mu, sd, n)
        survival = np.concatenate([np.ones((n, 6, 1)),
            expit(aa[None, :, None] * (theta[:, None, None] - bb[None, :, :])), np.zeros((n, 6, 1))], axis=2)
        probabilities = survival[:, :, :-1] - survival[:, :, 1:]
        parts.append((rng.random((n, 6, 1)) > np.cumsum(probabilities, axis=2)).sum(axis=2))
    return (pd.DataFrame(np.vstack(parts), columns=names), pd.Series(["R"] * n_reference + ["F"] * n_focal),
            dict(a=np.stack([a, af]), b=np.stack([b, bf]), mu=-.4, sd=1.2))


def compare_reference(condition="both", seed=20261024, points=61, hypotheses=None):
    """Independent Python mirt EM calibration; never reuse its DIF p-values."""
    import importlib.metadata
    from mirt.multigroup import fit_multigroup, InvarianceSpec
    from core import grm_dif as G
    x, groups, _ = generate(seed, condition)
    inputs = G.validate_inputs(x, groups, "R", "F", list(x.columns[:4]))
    fitter = G.Fitter(inputs, points)
    output = []
    start = None
    for name, roles in (("H0", ["anchor"] * 5 + ["ab"]),
                        ("Hb", ["anchor"] * 4 + ["b", "ab"]),
                        ("Ha", ["anchor"] * 4 + ["a", "ab"]),
                        ("H2", ["anchor"] * 4 + ["ab", "ab"])):
        if hypotheses is not None and name not in hypotheses:
            continue
        native = fitter.fit(roles, start)
        start = native
        spec = InvarianceSpec("scalar", free_discrimination=[i for i, role in enumerate(roles) if role in ("a", "ab")],
                              free_intercepts=[i for i, role in enumerate(roles) if role in ("b", "ab")])
        external = fit_multigroup(x, groups.to_numpy(), model="GRM", invariance=spec,
                                  n_categories=3, reference_group="R", n_quadpts=points, max_iter=3000, tol=1e-9)
        indices = [external.group_labels.index(label) for label in ("R", "F")]
        params = [external.model.get_group_parameters(i) for i in indices]
        latent = external.latent_distributions[indices[1]]
        a = np.stack([p["discrimination"] for p in params])
        b = np.stack([p["thresholds"] for p in params])
        encoded = native["design"].encode(a, b, float(latent.mean[0]), float(np.sqrt(latent.cov[0, 0])))
        value, gradient = G.objective(encoded, native["design"], fitter.parts, fitter.grid)
        native_a, native_b, mu, sd = native["design"].unpack(native["params"])
        output.append(dict(hypothesis=name, native_converged=native["converged"], external_converged=external.converged,
            native_parameters=native["design"].size, external_parameters=external.n_parameters,
            native_loglik=native["loglik"], external_loglik=external.log_likelihood,
            recomputed_external_loglik=-value * len(x), external_max_score=float(np.max(np.abs(gradient))),
            max_slope_difference=float(np.max(np.abs(native_a - a))),
            max_threshold_difference=float(np.max(np.abs(native_b - b))),
            focal_mean_difference=mu - float(latent.mean[0]), focal_sd_difference=sd - float(np.sqrt(latent.cov[0, 0])),
            external_a=a.tolist(), external_b=b.tolist(),
            external_focal_mean=float(latent.mean[0]), external_focal_sd=float(np.sqrt(latent.cov[0, 0]))))
    return dict(condition=condition, seed=seed, points=points, model="GRM",
                external_engine="Python mirt (NOT R mirt)", external_version=importlib.metadata.version("mirt"),
                results=output)


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--condition", default="both", choices=["null", "threshold", "slope", "both"])
    args = parser.parse_args()
    result = compare_reference(args.condition)
    args.reference_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(args.reference_output), "models": len(result["results"])}))
