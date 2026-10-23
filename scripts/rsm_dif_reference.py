"""Independent RSM reference: direct respondent likelihood + numerical derivatives.

No native Design/objective/score/pattern compression in this calibration. This is
a transparent analytic/numerical benchmark, NOT an R mirt calibration.
"""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.polytomous_dif_validation import generate, ANCHORS


def compare_reference(condition="both", seed=20261025):
    from core import polytomous_dif as P
    x, groups, _ = generate("RSM", seed, condition)
    responses = [x.loc[groups == g].to_numpy() for g in ("R", "F")]
    nodes, weights = np.polynomial.hermite.hermgauss(101)
    nodes, lw = nodes * np.sqrt(2), np.log(weights / np.sqrt(np.pi))
    native = P.Fitter(P.validate_inputs(x, groups, "R", "F", ANCHORS, "RSM"), 61)
    output, start = [], None
    for h, free in (("H0", [5]), ("H2", [4, 5])):
        # Independent physical coordinates: reference locations, free focal
        # locations, centered common tau1, focal mu and SD (not log SD).
        def decode(p):
            beta = np.stack([p[:6], p[:6]])
            beta[1, free] = p[6:6 + len(free)]
            tau = np.array([p[-3], -p[-3]])
            return beta, tau, p[-2], p[-1]
        def objective(p):
            beta, tau, mu, sd = decode(p)
            total = 0.
            for g, y in enumerate(responses):
                theta = nodes if g == 0 else mu + sd * nodes
                increments = theta[:, None, None] - beta[g][None, :, None] - tau[None, None]
                eta = np.concatenate([np.zeros((len(nodes), 6, 1)), np.cumsum(increments, axis=2)], axis=2)
                lp = eta - logsumexp(eta, axis=2, keepdims=True)
                joint = sum(lp[:, i, y[:, i]] for i in range(6)).T + lw
                total += logsumexp(joint, axis=1).sum()
            return -total / len(x)
        p0 = np.r_[np.zeros(6 + len(free)), -.5, 0., 1.]
        fit = minimize(objective, p0, method="L-BFGS-B", bounds=[(-8, 8)] * (6 + len(free)) + [(-4, 4), (-3, 3), (.3, 3)],
                       options=dict(maxiter=3000, ftol=1e-14, gtol=1e-8, maxls=50))
        beta, tau, mu, sd = decode(fit.x)
        A = np.broadcast_to(np.arange(3, dtype=float), (2, 6, 3)).copy()
        D = np.concatenate([np.zeros((2, 6, 1)), -np.cumsum(beta[:, :, None] + tau, axis=2)], axis=2)
        roles = ["ab" if i in free else "anchor" for i in range(6)]
        f = native.fit(roles, start); start = f
        NA, ND, nm, ns = f["design"].unpack(f["params"])
        output.append(dict(hypothesis=h, external_converged=bool(fit.success), native_converged=f["converged"],
            external_loglik=-fit.fun * len(x), native_loglik=f["loglik"],
            external_parameters=len(fit.x), native_parameters=f["design"].size,
            external_A=A.tolist(), external_D=D.tolist(), external_focal_mean=float(mu), external_focal_sd=float(sd),
            max_A_difference=float(np.max(abs(A - NA))), max_D_difference=float(np.max(abs(D - ND))),
            focal_mean_difference=nm - mu, focal_sd_difference=ns - sd, external_score=float(np.max(abs(fit.jac)))))
    return dict(model="RSM", condition=condition, seed=seed, points=61, external_points=101,
                external_engine="Independent direct RSM likelihood, centered steps, numerical L-BFGS-B, physical SD; NOT R mirt",
                results=output)


if __name__ == "__main__":
    for condition in ("null", "both"):
        result = compare_reference(condition)
        path = Path("tests/fixtures/polytomous_dif") / f"independent_RSM_{condition}.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(condition, [{k: r[k] for k in ("hypothesis", "external_converged", "max_D_difference", "focal_mean_difference", "focal_sd_difference")} for r in result["results"]], flush=True)
