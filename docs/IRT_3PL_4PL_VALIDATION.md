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

## Further internal verifications and reproducible commands

- `tests/test_irt_3pl4pl_marginal_gradient.py`: independent finite differences
  of the marginal log-likelihood verify the analytic 3PL/4PL score.
- `tests/test_irt_3pl4pl_mathematical_consistency.py`: independent check of
  ICC information, marginal log-likelihood, AIC and BIC identities.
- `core/irt_stationarity.py`: per-parameter first-order KKT necessary
  condition (gradient per examinee; descriptive threshold, not proof).
- `core/irt_truth_scenarios.py`: five explicitly bounded parameter regimes.
- `core/irt_quadrature_validation.py`: different estimation quadratures
  compared on one common high-resolution likelihood grid.
- `core/irt_se_coverage.py`: report *availability* of Hessian SE before
  conditional interval coverage, including all declined runs.
- `core/irt_model_reliability.py`: qualitative diagnostic status; never
  equates convergence with validation.

For complete deterministic CI tests, review the standard GitHub Actions
`PsikoAnaliz tests` workflow. Scientific evidence is stored separately
under `IRT benchmark (opt-in)`, `IRT multistart empirical check`, and
`IRT 3PL 4PL internal scientific audit`.

The early paired initialization experiment is stored in
[GitHub Actions run 39000776556](https://github.com/Tatung40/psikoanaliz/actions/runs/39000776556).
For 4PL N=500 with two replications, three initializations changed the
convergence count from 1/2 to 2/2, but required about three times the
computation and did not establish lower parameter error. Therefore one
or three starts remain user-selectable.

None of these internal studies, even if all workflows succeed, substitutes
for independent implementation equivalence or full Monte Carlo evidence
with quantified Monte Carlo sampling error.

## Extended matched-generating-model Monte Carlo (10 replications per N)

Completed [GitHub Actions run 39001046381](https://github.com/Tatung40/psikoanaliz/actions/runs/39001046381)
with N=200,500,1000,2000; 25 items; 10 replications for *each*
generating model and sample size; maximum 70 EM iterations, 21
Gauss-Hermite points, single default start. Results are conditional
on converged calibrations.

| N | Correct 3PL convergence | 3PL MAE(a) | 3PL mean boundary parameters | Correct 4PL convergence | 4PL MAE(a) | 4PL mean boundary parameters |
|---|---:|---:|---:|---:|---:|---:|
| 200 | 10/10 | 0.544 | 12.6 | 10/10 | 1.162 | 31.2 |
| 500 | 10/10 | 0.311 | 7.4 | 9/10 | 0.857 | 22.3 |
| 1000 | 10/10 | 0.188 | 6.2 | 7/10 | 0.704 | 16.1 |
| 2000 | 10/10 | 0.153 | 4.6 | 10/10 | 0.452 | 15.7 |

Across each generative branch 80 model fits were attempted because both
3PL and 4PL were fitted to each simulated dataset; 79/80 and 76/80
converged respectively, including misspecified model fits. The table
above reports only *matched* generating/fitted model pairs. Ten draws
per condition remain exploratory and cannot establish nominal statistical
coverage, general robustness or universal bias properties.

The [independent internal 3PL/4PL audit](https://github.com/Tatung40/psikoanaliz/actions/runs/39001405713)
tested a five-item N=500 stress case. Neither model supplied a
publishable Hessian CI: 3PL SE was refused in both runs and 4PL
calibration did not converge in both runs. The automated refusal guards
correctly prevented presenting these as valid intervals.

**Final scientific boundary:** software regression tests and internal
numerical identities are passed, but neither an external real-data
calibration comparison nor broad SE interval coverage studies have been
completed. This is an implementation validation stage, not a claim that
3PL/4PL are fully scientifically validated.

## External comparison with R mirt 1.48 (B3, fixed-parameter and 3PL estimation)

Reproduce: `python scripts/make_mirt_fixtures.py`, then
`Rscript scripts/mirt_reference_3pl4pl.R` (writes `tests/fixtures/mirt/mirt_results.csv`
and `mirt_session.txt`; R 4.3.3, mirt 1.48 built from source commit
a955235, SimDesign 2.24, Deriv 4.2.0). Checked by `tests/test_irt_mirt_reference.py`.

Matched conditions: mirt `P=g+(u-g)/(1+exp(-(a1*theta+d)))` with `a=a1`, `b=-d/a1`,
`c=g`, `d=u`; g and u are stored on the logit scale inside mirt, so derivatives are
mapped with the Jacobian including its second-order term. Integration uses the same
21-node Gauss-Hermite grid (`customTheta`, `customPriorFun`). Fixtures: 3PL k=4 and
4PL k=5 (N=400 and 2000), and a 3PL k=8, N=3000 free-estimation case.

| Quantity | Max. relative difference | Status |
|---|---:|---|
| Item response function (absolute) | 5.6e-16 | verified |
| Marginal log-likelihood (GH-21) | 2.6e-15 | verified (3PL, 4PL) |
| Analytic score vs mirt numerical gradient | 2.7e-9 | verified (3PL, 4PL) |
| Observed information vs Hessian of mirt logLik | 6.6e-8 | verified (3PL, 4PL; also indefinite cases) |
| Hessian SE vs mirt (3PL, positive definite case) | 7.7e-8 | verified |
| mirt `SE.type='Richardson'`, default 61-node grid | info 6.0e-8, SE 4.7e-8 | verified (3PL) |
| SE refusal decision (not positive definite) | identical in 4/4 cases | verified |
| Free 3PL ML estimates | <=0.0014 SE; logLik gap 1.5e-6 | verified (one interior case) |

Not verified: free 4PL estimation (no small interior 4PL MLE was found; Python's
bounded space and mirt's logit space then target different points), Oakes/Louis/
sandwich SE types, real data, other quadrature sizes. Note: mirt's numerical
SE.type Hessian re-weights custom nodes by `dnorm(theta)` and ignores
`customPriorFun` (mirt 1.48, `R/utils.R::BL.LL`), so that route is compared on
mirt's default grid only.

Structural identifiability (resolved in B4): complete 0/1 data on k items give
`2**k-1` pattern degrees of freedom. `fit_irt`, `fit_irt_multistart`, the Hessian
SE and the OPG approximation now reject models with more free parameters
(3PL with k=3: 9 > 7; 4PL with k=3 or 4: 12 > 7, 16 > 15), like mirt. The rule
is a necessary condition only and lives in `core.irt.structural_identifiability`,
not in the model-independent `validate_binary`. Item priors would not remove the
problem. In B2 a 4PL with k=4 passed the numerical SE gates (condition number
about 4e3), which is why the structural refusal is checked first. No identifiable
4PL fixture (k=5..8, N<=4000) gave a positive definite information matrix at the
generating values.

## Convergence diagnostics (B4)

`converged` keeps the marginal log-likelihood criterion. `fit["convergence"]`
also reports the largest parameter change in the last EM cycle (default
diagnostic threshold `param_tolerance=1e-3`), split into interior parameters and
parameters on an optimization bound, with status `converged`, `loglik_only`
(warned, not rejected) or `not_converged`. On the interior 3PL fixture the
default logL stop lies within 0.022 SE of the mirt MLE, so the parameter
criterion is reported as a warning rather than used to reject estimates.

## Free 4PL estimation vs R mirt 1.48 (B5)

Reproduce: `python scripts/make_mirt_fixtures.py`; `Rscript scripts/mirt_free_4pl.R`
(~3 min); `python scripts/python_free_4pl.py` (~5 min). Checked by
`tests/test_irt_mirt_free4pl.py` (~3 s; `PSIKOANALIZ_SLOW_TESTS=1` adds a live k=5
optimization and a full k=8 refit, ~50 s). Data: 4PL, k=5 and k=8, N=3000, seed 2026.
Matched: data, parameterization, N(0,1) fixed (recorded), GH-21 nodes/weights, all
a, b, c, d free, no item priors (recorded `prior.type = none`), the three Python
starting values. mirt stops on max |parameter change| < 1e-7, Python on |dlogL| < 1e-9;
solutions are compared through the same likelihood, not through the stopping rules.

| Check | Result | Class |
|---|---|---|
| logLik at all 8 mirt solutions (incl. degenerate ones) | rel. diff <= 2.6e-15 | PASS |
| Gradient at 8 mirt solutions, mirt internal space (a1, d, logit g, logit u) | abs. diff <= 1.1e-7 | PASS |
| Observed information at unbounded mirt solutions, internal space | rel. diff <= 1.8e-7; both numerically singular | PASS |
| Unbounded mirt optimum | not a finite interior MLE: one slope runs to a = 22-166 depending on start (k=5), a = 13-19 (k=8), logLik differs < 1.2e-3; lies outside Python's box (a <= 4) | WARNING (not comparable) |
| Bounded optimum (Python box; mirt with the same lbound/ubound) | mirt and all Python solutions, maximized further by L-BFGS-B on the validated likelihood, reach one KKT point (KKT <= 4e-8 per person): logLik equal within 1e-5, parameters within 1e-3 | PASS |
| How close each EM stops to that optimum | k=8: Python logLik gap 1e-5, max |da| 0.009 (converged); mirt-box gap 9e-4, |da| 0.006. k=5: Python gap 5e-4 to 4.6e-3, max |da| 0.12-0.31, not converged after 5000 cycles; mirt-box gap 1.8e-3, |da| 0.034 | WARNING (slow EM on a flat surface) |
| SE decision | optimum has 4 (k=5) / 8 (k=8) parameters on bounds: Python refuses SE; mirt's unbounded information is singular: no reportable SE in either | PASS |
| Multistart stability | Python k=8: logLik spread 1e-6, params 5e-4; k=5: logLik spread 4e-3, a spread 0.19 (EM unfinished), but all starts reach the same bounded optimum after direct optimization | PASS (optimum) / WARNING (EM) |
| Recovery of generating values | max |a - a_true| 2.2 (k=5), 1.9 (k=8) at the bounded optimum | reported only; sampling/identifiability, not an implementation property |
| 4PL Hessian SE at an interior MLE | no interior finite 4PL MLE in these fixtures | NOT TESTED |

No reproducible mathematical or algorithmic disagreement with mirt was found (no FAIL).

## Estimation speed and optimizer stability (B6)

Profile (4PL k=5, N=3000, 300 EM cycles: 22.7 ms/cycle): 78% of the time is the
item-wise M-step L-BFGS-B, mostly its numerical gradient. The limiting factor is
the number of EM cycles on the flat 4PL surface (B5: 5000 cycles still short of the
optimum), so three approaches were compared on the B5 fixtures (same machine,
3 repeats for the shipped options):

| Case | Method | Time (s) | EM / opt. iterations | logL vs B5 optimum | max abs param diff | KKT/person | Status |
|---|---|---:|---:|---:|---:|---:|---|
| 4PL k=5 | EM default (tol 1e-4, 70) | 1.54 ± 0.06 | 70 / 0 | -0.18 | 1.21 | 9.6e-4 | not converged |
| 4PL k=5 | EM tight (B5 fixture, 1 run) | 48 | 5000 / 0 | -2.9e-3 | 0.25 | 5.5e-6 | not converged |
| 4PL k=5 | safeguarded SQUAREM (prototype, 1 run) | 14.8 | 1500 EM evals | -2.6e-3 | 0.24 | 3.1e-5 | not converged |
| 4PL k=5 | **EM + L-BFGS-B** | 4.74 ± 0.20 | 70 / 658 | -4.8e-9 | 5.8e-4 | 7.7e-8 | converged (KKT) |
| 4PL k=8 | EM default | 2.19 ± 0.05 | 70 / 0 | -0.10 | 0.21 | 3.7e-4 | not converged |
| 4PL k=8 | EM tight (B5 fixture) | 37.5 | 2988 / 0 | -1.0e-5 | 9.1e-3 | 7.2e-7 | converged |
| 4PL k=8 | safeguarded SQUAREM (prototype) | 18.8 | 1500 EM evals | -5.1e-5 | 0.021 | 3.8e-6 | not converged |
| 4PL k=8 | **EM + L-BFGS-B** | 4.53 ± 0.16 | 70 / 481 | -1.1e-8 | 7.9e-5 | 6.0e-8 | converged (KKT) |
| 3PL k=8 (vs mirt MLE) | EM default | 0.61 ± 0.08 | 25 / 0 | -8.4e-4 | 8.1e-3 | - | converged |
| 3PL k=8 (vs mirt MLE) | **EM + L-BFGS-B** | 1.51 ± 0.06 | 25 / 175 | -1.4e-10 | 4.3e-6 | 3.4e-8 | converged (KKT) |

SE decisions are unchanged: 4PL optima have 4 (k=5) / 8 (k=8) parameters on bounds,
so SEs are refused; the interior 3PL solution gets SEs. SQUAREM gave no measurable
benefit (frequent safeguard fallbacks, no convergence in 1500 evaluations) and is not
shipped.

Decision: `fit_irt(..., refine=True)` (3PL/4PL, opt-in; default EM unchanged). After
EM the same marginal logL (same clipping and GH grid) is maximized in the same bounds
by L-BFGS-B with the analytic score. The result is accepted only if finite, inside
the bounds and not lower in logL; the optimizer's success flag is recorded but not
trusted. Otherwise the EM solution is kept and a warning states why. KKT violation is
reported before and after (`fit["refinement"]`); `converged` becomes True after an
accepted refinement only when KKT <= `kkt_tolerance` (default 1e-6 per person), with
`convergence["criterion"] = "kkt_after_refinement"`. Reproduce:
`python scripts/irt_refine_benchmark.py`; tests: `tests/test_irt_refinement.py`.
