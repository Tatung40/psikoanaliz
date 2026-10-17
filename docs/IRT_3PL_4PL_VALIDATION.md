# 3PL / 4PL scientific validation record

**Status:** Internal verification and simulations are available. **NOT independently
validated against an external calibration implementation under matched fixed
parameters and real data.** The software must not claim otherwise.

## Model and numerical definition

- Dichotomous unidimensional logistic IRT:
  `P_i(theta)=c_i+(d_i-c_i) sigmoid[a_i(theta-b_i)]`.
- Identification: examinee ability prior N(0,1), fixed. 3PL fixes d=1.
- Estimation: Gauss-Hermite marginal maximum likelihood, EM, L-BFGS-B
  itemwise M step, bounded parameter space.
- Original default limits: a in [0.15,4], b in [-4,4], c in [0,.35],
  4PL d in [.65,1]. Expanded limits are exploratory, not recommended
  as a fix for identifiability.
- Convergence: marginal log-likelihood increment under configured tolerance;
  nonconvergence is reported and never interpreted as proven reliability.
- Three deterministic EM starting positions can be compared; selected converged
  candidate maximizes the fitted marginal log-likelihood.
- 3PL and 4PL with many parameters can remain practically unidentifiable,
  even when the EM stopping rule fires.

## What has been independently checked internally

- Data generation reproducibility, probability frequencies, and parameter
  admissibility (automated deterministic tests).
- Fitted solution selection across starting points and per-start audit.
- Repeated-simulation design, seed tracking, empirical bias, MAE and RMSE.
- Paired (same observed data) one-start versus three-start benchmark.
- Boundary proximity flags and conservative model-reliability interpretation.
- Refusal to claim Hessian standard errors on nonconverged fits.
- Hessian observed information numerical tests for 1PL/2PL. **This is
  NOT evidence of 3PL/4PL SE equivalence to external software.**

## Preliminary simulation results (NOT publication-grade)

From [2026-10-17 paired generating-model simulation](https://github.com/Tatung40/psikoanaliz/actions/runs/38999527120):
25 items, data generated under the correct model, three replications per
sample size, Gauss-Hermite with 21 nodes, max 70 EM iterations.

| N | 3PL correct-model MAE(a) | 4PL correct-model MAE(a) |
|---|---:|---:|
| 200 | 0.606 | 1.143 |
| 500 | 0.316 | 0.883 |
| 1000 | 0.243 | 0.746 |
| 2000 | 0.153 | 0.405 |

MAE improvements with N are promising but no uncertainty intervals are
justifiable at three replications. The 4PL has substantial boundary and
identification issues; do not interpret mere convergence as correctness.
Mismatched fitted and generating models must be labeled separately.

## Required before externally validated / publication-ready designation

1. Fixed-parameter comparison of 3PL and 4PL log-likelihood, gradients,
   model fit, and covariance / standard errors with independent R mirt,
   or another rigorously verified scientific implementation.
2. Full Monte Carlo design with multiple truth regimes:
   a, b, c, d ranges, ability distributions, sample sizes, item counts,
   and at least dozens (preferably 100+) successful and failed replications
   separately accounted for in Monte Carlo standard errors.
3. Coverage assessment of 3PL/4PL SE intervals against known true item
   parameters, including near-boundary scenarios and failure rates.
4. Sensitivity to quadrature nodes, EM iteration limits, tolerance,
   initialization, and alternative optimization paths.
5. Runtime/memory under concurrent Streamlit users and the constrained
   free host. Avoid heavy simulations in the interactive request thread.
6. Documentation: no automatic choice of the most complex model; model
   fit, local dependence, information-matrix eigenvalues and boundary
   diagnostics need triangulation.

**Do not remove the experimental 3PL/4PL SH warning solely because unit
tests or internal simulations pass.**
