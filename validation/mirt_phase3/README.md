# Phase 3: general CFA dichotomous validation prototype

## Scope and safety

Python mirt 1.2.0 rejects multidimensional 3PL/4PL. No guard is patched and
no new capability is attributed to that package. The user explicitly authorized
an independent Python scientific component where the public API is insufficient.
`engine.py` is a NEW SciPy validation prototype, not the production estimator;
no UI/model registration changes are made before independent acceptance.
Prior PR80/81 estimators, Phase2 exact compositions and inference gates remain
unchanged. Correlated 2PL Stage A reuses actual accepted R calibration evidence
in stage_A_evidence.json; no accepted profile is refitted.

The prototype directly maximizes normalized fixed-GH marginal likelihood.
P=c+(u-c)*logistic(A theta+delta), D=1. Named CFA axes have >=3 pure positive
anchors each, zero masked loadings, fixed latent mean0 and variances1.
Full covariance uses L lower triangular/unit diagonal, C=LL'/sqrt(diag products);
eta=lower(L) in [-3,3]. All such C are positive definite. Positive active slopes
[.1,5], intercepts [-6,6], c [0,.5], u [.5,1]. This restricts supported loading
signs/correlation parameters explicitly; it is not unrestricted rotational MIRT.
The corner c=u=.5 is invalid and rejected, never silently clipped. Free 4PL
endpoint/strict-order feasibility needs separate runtime evidence.

ML/no priors, fixed GH15 per axis (225/3375 nodes), covariance-dependent normal
density-ratio masses normalized identically in R customGroup and Python. This
is finite-grid numerical equivalence, not quadrature-accuracy certification.
Analytic physical item/covariance likelihood scores are checked independently
at two finite-difference steps; information matrices against theta derivatives.
The R reference uses native multidimensional 3PL/4PL mirt estimation, not the
Python likelihood for fitting. Native R probabilities, information and logLik
are retained. EAP/SD are conditional finite-common-grid posterior quantities;
MAP, parameter SE/CI, formal LR/p and sampling coverage remain unvalidated.

## Staging and reproducibility

Only fixed-c 3PL is initially requested: 2D correlated between-item and 3D
correlated within-item. Actual existing PR80 2PL binary fixtures are reused,
with no new response simulation. Thus these are cross-engine optimization
benchmarks, not recovery or model-fit studies of true 3PL populations.
Known c=.2, u=1; starting active A=1.2, delta0, eta0. Criteria are reused
UNCHANGED from validation/mirt_stage1/criteria.json, with hashes in reports.
SciPy L-BFGS-B ftol1e-13/gtol1e-7/max2000 vs R EM maximum parameter change
1e-8/max3000 and SLSQP xtol1e-8. Native stopping messages, full physical and
projected scores, active bounds, two-step tangent observed-Hessian spectra and
condition diagnostics are reported separately. No scientific PASS follows
from the optimizer success flag. Two converged engines + comparable model +
all preregistered comparisons + valid constraints are required for numerical
acceptance; no scientific certification is claimed.

Command, from repo root with R mirt 1.48 installed:
`PYTHONPATH=. python validation/mirt_phase3/compare.py --output validation-output/mirt-phase3`.
Existing independent R workflow/cache is reused. One actual Python and R fit
per explicitly requested profile, no retries/alternate starts in the harness.
Preflight probability/information/prior mappings must pass BEFORE calibration.
Failure stops the dependent stage, preserves inputs, versions, hashes, logs,
parameter values and diagnostic reports, and fails CI visibly. R unavailable
means BLOCKED / NOT EXECUTED, never PASS. No R validation executed locally.

Current status: Stage A retained NUMERICALLY VALIDATED / NOT CERTIFIED;
new general fixed/free 3PL/4PL configurations PENDING actual R calibration.
4D/5D explicitly UNSUPPORTED here. No new production capabilities claimed.
