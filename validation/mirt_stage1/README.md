# Phase 1 — independent 2D/3D 2PL and GRM comparison

Status: **STAGE 1 NUMERICALLY VALIDATED / NOT CERTIFIED** (run 40383677441; four profiles converged in both engines). Stage 2 follows only after this recorded acceptance. Local R is unavailable; actual R execution is delegated to the dedicated PR workflow, with all outputs retained. An API probe or fixed-parameter preflight is not a converged calibration.

Four profiles use N=400, 4 pure indicators per factor plus 2 additional indicators. 2PL 2D and GRM 3D estimate only the F1–F2 correlation; remaining correlations stay zero. 2PL 3D / GRM 2D have within-item loadings on the extra items and orthogonal factors. Labels, seeds, exact inputs and hashes are recorded per case. These restricted profiles do not establish general exploratory/bifactor or arbitrary 3D covariance support.

Both engines use ML, no priors, unit variances, zero means, identified CFA patterns, positive active loadings in [.1,5], fixed structural zeros, common preregistered starts, GH15 per axis (225 / 3375 nodes), max_iter=1000 and tol=1e-7. Python's EM stopping statistic and R's parameter-change criterion differ and are recorded; numerical agreement requires actual convergence and all preregistered comparisons, not the matching name of a tolerance argument.

2PL uses slope/intercept coordinates. GRM uses Python physical thresholds b and R d=-sum(a)*b. Bounds b in [-6,6] and adjacent gaps >=1e-6 become exact linear inequalities in R slope/intercept coordinates. The actual R mirt_df/data.frame class and native attributes are preserved. Correlated GH masses require density-ratio reweighting. R's ordinary customPriorFun has no current covariance argument; the public createGroup Gaussian density uses a Fisher-coordinate F1–F2 correlation with GH/reference-density weights. This mapping and prior masses must pass the actual fixed-parameter R preflight **before any calibration**. It is not legitimate to silently compare R's ordinary equally spaced/default-normal weights with Python GH masses.

All parameterizations, constraint feasibility, category probabilities and information matrices are checked before fitting. A failed preflight stops that profile's calibration. There are no automatic retries or alternate starts. Failures remain failures in report.json and raw R logs. For converged fits compare transformed parameters, covariance, probabilities, full item information matrices, first 16 EAP/posterior SD results and marginal logLik on the common grid; preserve native logLik/AIC/BIC/iterations. R EAP reference is explicit posterior summation using actual R item probabilities, not a claim to R fscores execution. Prediction surfaces and quadrature nodes are separate grids. No unverified parameter SE/CI/p is returned (multidimensional auto SE uses complete-data curvature).

`criteria.json` fixes strict numerical reproducibility targets before fitting, with units and rationale. Passing means numerical validation of that finite-grid profile only; it does not certify population bias, coverage, size, optimality or quadrature accuracy. Boundary/weak-identification and quadrature-refinement evidence are still required before broader scientific certification. No thresholds may be adjusted to fit observed differences.

```bash
PYTHONPATH=. python -m pytest -q tests/test_mirt_stage1.py
PYTHONPATH=. python validation/mirt_stage1/compare.py --output validation-output/mirt-stage1
```

The primary engine is always the existing Python mirt public API. No new calibration algorithm is introduced. Next executable step: run the independent R workflow on the draft PR; inspect per-profile mapping, convergence, numerical discrepancies and retained logs before enabling the UI or further model families.

## Observed run 40381541591 (preserved)

Actual independent calibration converged for orthogonal 2PL 3D and GRM 2D. Their common-grid likelihood differences were 1.62e-7 / 3.26e-7, max slope differences .000205 / .000065, max probability differences .0000132 / .0000227; EAP, posterior SD and information also met the unchanged preregistered numerical criteria. The final native-R likelihood check failed because jsonlite's default four-decimal serialization rounded metadata, not because the actual likelihoods disagreed. Correlated preflights stopped on an unnecessary missing `mvtnorm` import. Both original reports and errors remain in the run artifact.

Follow-up preserves the successful calibrations: download the original 534KB artifact, reconstruct fixed R objects from their saved native parameter values (all estimation flags FALSE), and reevaluate native likelihood at full JSON precision. No second calibration of these profiles. The original failed report is copied unchanged to `original_report.json`. Correlated profiles alone execute their first calibration after a base-R Gaussian density-ratio preflight. The density-ratio equation is tested independently against SciPy; no new package or tolerance change is needed. Source-free-parameter count and fixed-evaluation provenance are retained explicitly.

## Expansion after actual Stage 1 acceptance

`phase1_evidence.json` retains the four actual converged R/Python reports.
`--expand` reuses those estimates without recalibration and adds eight seeded
profiles: unit-slope Rasch, unit-slope GPCM/PCM, correlated 3D and orthogonal
5D GPCM, full nominal 2D, orthogonal 4D bifactor, and identified nonnegative
triangular 2D exploratory 2PL/GRM. The exploratory scope is restricted; it is
not unrestricted signed EFA. Higher-dimensional quadrature is coarse and is
not a quadrature-accuracy certificate. Bifactor full tensor and reduced
integration represent the same finite-grid sum. No model is certified.

Full multidimensional NRM uses R native `gpcm_mats` with fixed unit item
slopes and freely estimated category-by-factor scores (reference category
zero), not R's rank-one nominal item. Nominal information is calculated from
R native probabilities with two independently checked derivative steps;
standard-score native `iteminfo` is unsuitable for this representation.
Rasch/PCM use fixed unit loadings, not shared-free-discrimination 1PL.
3PL/4PL and RSM remain disabled: the installed public API does not expose
an equivalent multidimensional family/constraint. All previous criteria
remain unchanged; preflight failure prevents a fit, and nonconvergence fails.

## User-facing safety

The separate MIRT route calls public library APIs only. All new analyses are
EXPERIMENTAL / NOT CERTIFIED. The actual full dataframe contents, IDs, item
order, model, matrix, coding, covariance mode and solver settings enter the
same existing deterministic fingerprint used by Python-mirt DIF. Changes
invalidate every result/download before rendering. Nonconvergence exposes
only provisional parameters/diagnostics, never scores, information or plots.
CSV/XLSX use the existing spreadsheet-safe exporter; ZIP includes all tables,
metadata, method/limitations and PNG. No formal model-fit, SE/CI/p or nominal
Wright Map is manufactured. Public FitResult lacks optimizer gradients and
iteration history; no stationarity or global-optimum claim is made. MAP has
no displayed unverified SE; only EAP posterior SD and full-matrix Fisher SEM
are shown. Fisher inversion is suppressed unless the smallest eigenvalue
exceeds sqrt(machine epsilon) times max(largest eigenvalue,1), a numerical
stability guard rather than a scientific acceptance threshold. Parameter-bound
flags are numerical diagnostics, not inference.

## Actual expansion run and one diagnosed continuation

Run 40394398251: seven CFA profiles (including Rasch, PCM, GPCM 3D) passed
unchanged numerical criteria with both engines converged. Full original
expansion reports are retained in `expansion_original_evidence.json` and the
3.9MB workflow artifact. Bifactor and triangular 2PL reached 1000 iterations
without joint convergence; GPCM 5D and triangular GRM missed numerical
criteria. None of these results is relabelled PASS.

NRM fixed probabilities/information agreed, but R's fitted score matrices
never moved from their starts although marked free. R mirt 1.48
[`src/dpars.cpp`, `d_nominal2`](https://github.com/cran/mirt/blob/1.48/src/dpars.cpp#L982)
initializes the derivative vector to zero and fills only item slopes and
intercepts, omitting category-score derivatives. Saved-solution central
common-grid likelihood derivatives at steps 1e-4 / 5e-5 agree and reach 7.90076
in a free score coordinate (`nrm_original_score_diagnostic.json`). Thus its
analytic-gradient convergence flag does not establish full NRM ML convergence.
The ordinary ordinal GPCM fixes these score coefficients and is unaffected.

A single opt-in `--refine-failed` continuation uses the complete saved Python
item solution as the same explicitly recorded initialization in both engines;
latent covariance starts at identity in both. The data, GH grids, bounds,
priors and all acceptance criteria remain unchanged. Outer stopping is
**tightened** from 1e-7 to 1e-9, with a declared ceiling of 3000 iterations,
addressing finite-stopping differences without relaxing any acceptance target.
For NRM only, R uses its supported NLopt BOBYQA optimizer, which uses the
native likelihood and bounds and ignores the incomplete analytic gradient;
other R models retain SLSQP. Inner xtol_rel=1e-8 / maxeval=1000 are unchanged.
Python estimators and application defaults remain unchanged. This is a
warm-start numerical comparison, not evidence of global optimality or
population-level coverage. Original failed reports and settings are retained
before continuation. Already accepted profiles are never recalibrated.

Local full regression: **1088 passed, 10 pre-existing skipped** (1255.89s).
Focused adapter/refinement tests and actual Streamlit checks cover the final
validation-only continuation and report controls separately.
