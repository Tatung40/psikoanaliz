# Phase 1 — independent 2D/3D 2PL and GRM comparison

Status: **PENDING / NOT CERTIFIED**. No Stage 2 models or public MIRT UI are enabled until Phase 1 completes. Local R is unavailable; actual R execution is delegated to the dedicated PR workflow, with all outputs retained. An API probe or fixed-parameter preflight is not a converged calibration.

Four profiles use N=400, 4 pure indicators per factor plus 2 additional indicators. 2PL 2D and GRM 3D estimate only the F1–F2 correlation; remaining correlations stay zero. 2PL 3D / GRM 2D have within-item loadings on the extra items and orthogonal factors. Labels, seeds, exact inputs and hashes are recorded per case. These restricted profiles do not establish general exploratory/bifactor or arbitrary 3D covariance support.

Both engines use ML, no priors, unit variances, zero means, identified CFA patterns, positive active loadings in [.1,5], fixed structural zeros, common preregistered starts, GH15 per axis (225 / 3375 nodes), max_iter=1000 and tol=1e-7. Python's EM stopping statistic and R's parameter-change criterion differ and are recorded; numerical agreement requires actual convergence and all preregistered comparisons, not the matching name of a tolerance argument.

2PL uses slope/intercept coordinates. GRM uses Python physical thresholds b and R d=-sum(a)*b. Bounds b in [-6,6] and adjacent gaps >=1e-6 become exact linear inequalities in R slope/intercept coordinates. R mirtValues class is preserved. Correlated GH masses require density-ratio reweighting. R's ordinary customPriorFun has no current covariance argument; the public createGroup Gaussian density uses a Fisher-coordinate F1–F2 correlation with GH/reference-density weights. This mapping and prior masses must pass the actual fixed-parameter R preflight **before any calibration**. It is not legitimate to silently compare R's ordinary equally spaced/default-normal weights with Python GH masses.

All parameterizations, constraint feasibility, category probabilities and information matrices are checked before fitting. A failed preflight stops that profile's calibration. There are no automatic retries or alternate starts. Failures remain failures in report.json and raw R logs. For converged fits compare transformed parameters, covariance, probabilities, full item information matrices, first 16 EAP/posterior SD results and marginal logLik on the common grid; preserve native logLik/AIC/BIC/iterations. R EAP reference is explicit posterior summation using actual R item probabilities, not a claim to R fscores execution. Prediction surfaces and quadrature nodes are separate grids. No unverified parameter SE/CI/p is returned (multidimensional auto SE uses complete-data curvature).

`criteria.json` fixes strict numerical reproducibility targets before fitting, with units and rationale. Passing means numerical validation of that finite-grid profile only; it does not certify population bias, coverage, size, optimality or quadrature accuracy. Boundary/weak-identification and quadrature-refinement evidence are still required before broader scientific certification. No thresholds may be adjusted to fit observed differences.

```bash
PYTHONPATH=. python -m pytest -q tests/test_mirt_stage1.py
PYTHONPATH=. python validation/mirt_stage1/compare.py --output validation-output/mirt-stage1
```

The primary engine is always the existing Python mirt public API. No new calibration algorithm is introduced. Next executable step: run the independent R workflow on the draft PR; inspect per-profile mapping, convergence, numerical discrepancies and retained logs before enabling the UI or further model families.
