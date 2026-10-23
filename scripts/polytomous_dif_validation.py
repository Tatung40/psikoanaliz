"""Independent generators and pinned Python mirt EM benchmarks (not R)."""
import json
from pathlib import Path
import sys
import importlib.metadata
import numpy as np
import pandas as pd
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ANCHORS = ["M1", "M2", "M3", "M4"]


def generate(model="PCM", seed=20261025, condition="both", n_reference=600, n_focal=400):
    if model not in ("PCM", "GPCM", "RSM", "NRM") or condition not in ("null", "both"):
        raise ValueError("Unknown model/condition")
    rng = np.random.default_rng(seed)
    steps = np.linspace(-.6, .6, 6)[:, None] + np.array([.5, -.5])  # deliberately disordered
    slopes = np.linspace(.8, 1.4, 6) if model == "GPCM" else np.ones(6)
    if model == "RSM": steps = np.linspace(-.6, .6, 6)[:, None] + [-.6, .6]
    sf, bf = slopes.copy(), steps.copy()
    if condition == "both":
        bf[4] += .6
        if model == "GPCM": sf[5] -= .6
        else: bf[5] -= .7
    if model == "NRM":
        A = np.column_stack([np.zeros(6), [.8, 1.1, -.4, .5, .9, -.5], [-.7, .2, 1.4, 1.8, -.8, 1.2]])
        D = np.column_stack([np.zeros(6), np.linspace(-.5, .3, 6), np.linspace(.4, -.4, 6)])
        AF, DF = A.copy(), D.copy()
        if condition == "both": DF[4, 1] += .9; AF[5, 2] -= .9
        As, Ds = [A, AF], [D, DF]
    else:
        As = [s[:, None] * np.arange(3) for s in (slopes, sf)]
        Ds = [np.column_stack([np.zeros(6), -s[:, None] * np.cumsum(b, axis=1)]) for s, b in ((slopes, steps), (sf, bf))]
    parts = []
    for n, mu, sd, A, D in ((n_reference, 0., 1., As[0], Ds[0]), (n_focal, -.4, 1.2, As[1], Ds[1])):
        theta = rng.normal(mu, sd, n)
        # Independent canonical generator, not the native likelihood function.
        probs = softmax(theta[:, None, None] * A[None] + D[None], axis=2)
        parts.append((rng.random((n, 6, 1)) > np.cumsum(probs, axis=2)).sum(axis=2))
    return (pd.DataFrame(np.vstack(parts), columns=[f"M{i + 1}" for i in range(6)]),
            pd.Series(["R"] * n_reference + ["F"] * n_focal),
            dict(A=np.array(As), D=np.array(Ds), mu=-.4, sd=1.2))


def compare_reference(model="PCM", condition="both", seed=20261025, points=61):
    from mirt.multigroup import fit_multigroup, InvarianceSpec
    from core import polytomous_dif as P
    if model == "RSM": raise ValueError("RSM has no public multigroup factory; use independent constrained benchmark")
    x, groups, truth = generate(model, seed, condition)
    inputs = P.validate_inputs(x, groups, "R", "F", ANCHORS, model)
    fitter = P.Fitter(inputs, points)
    hypotheses = [("H0", ["anchor"] * 5 + ["ab"])]
    if model != "PCM": hypotheses += [("Hb", ["anchor"] * 4 + ["b", "ab"]), ("Ha", ["anchor"] * 4 + ["a", "ab"])]
    hypotheses += [("H2", ["anchor"] * 4 + ["ab", "ab"])]
    output, start = [], None
    for name, roles in hypotheses:
        native = fitter.fit(roles, start); start = native
        spec = InvarianceSpec("scalar", free_discrimination=[i for i, r in enumerate(roles) if r in ("a", "ab")],
                              free_intercepts=[i for i, r in enumerate(roles) if r in ("b", "ab")])
        external = fit_multigroup(x, groups.to_numpy(), model=model, invariance=spec, n_categories=3,
            reference_group="R", n_quadpts=points, max_iter=4000, tol=1e-9)
        indices = [external.group_labels.index(label) for label in ("R", "F")]
        params = [external.model.get_group_parameters(i) for i in indices]
        latent = external.latent_distributions[indices[1]]
        if model == "NRM":
            A = np.stack([p["slopes"] for p in params]); D = np.stack([p["intercepts"] for p in params])
        else:
            a = np.stack([p["discrimination"] for p in params]); b = np.stack([p["steps"] for p in params])
            A = a[:, :, None] * np.arange(3)
            D = np.concatenate([np.zeros((2, 6, 1)), -a[:, :, None] * np.cumsum(b, axis=2)], axis=2)
        mu, sd = float(latent.mean[0]), float(np.sqrt(latent.cov[0, 0]))
        if model == "NRM" and A[0, 0, 1] < 0: A = -A; mu = -mu
        encoded = native["design"].encode(A, D, mu, sd)
        value, gradient = P.objective(encoded, native["design"], fitter.parts, fitter.grid)
        NA, ND, nm, ns = native["design"].unpack(native["params"])
        output.append(dict(hypothesis=name, native_converged=native["converged"], external_converged=external.converged,
            native_parameters=native["design"].size, external_parameters=external.n_parameters,
            native_loglik=native["loglik"], external_loglik=external.log_likelihood,
            recomputed_external_loglik=-value * len(x), external_max_score=float(np.max(np.abs(gradient))),
            max_A_difference=float(np.max(np.abs(NA - A))), max_D_difference=float(np.max(np.abs(ND - D))),
            focal_mean_difference=nm - mu, focal_sd_difference=ns - sd,
            external_A=A.tolist(), external_D=D.tolist(), external_focal_mean=mu, external_focal_sd=sd))
    return dict(model=model, condition=condition, seed=seed, points=points,
                external_engine="Python mirt (NOT R mirt)", external_version=importlib.metadata.version("mirt"), results=output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["PCM", "GPCM", "NRM"], required=True)
    parser.add_argument("--condition", choices=["null", "both"], default="both")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(compare_reference(args.model, args.condition), indent=2), encoding="utf-8")
    print(args.output)
