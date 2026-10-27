# Phase 3: general CFA dichotomous validation prototype

## Scope and safety

Python mirt 1.2.0 rejects multidimensional 3PL/4PL. No guard is patched and
no new capability is attributed to that package. The user explicitly authorized
an independent Python scientific component where the public API is insufficient.
`engine.py` initially supplied a NEW isolated SciPy validation prototype;
only independently accepted scopes are promoted to the explicitly named
experimental production component below. No native-library capability is invented.
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

Stage A and subsequently accepted per-profile results are NUMERICALLY
VALIDATED / NOT CERTIFIED; the final scope and resource blockers are below.
4D/5D explicitly UNSUPPORTED here. No new production capabilities claimed.

## First actual execution and representation repair

Run 40490151345: native R fixed-parameter probability/information/prior mapping
passed (max errors 5.56e-16 / 4.86e-16 / 2.78e-16). The Python 2D fit
actually converged in 55 iterations, LL -2536.636921214727, projected score
3.31e-5. R fitting completed the call and wrote raw native estimates, then
failed the exact `all(diag(S)==1)` assertion. Native convergence status was
not yet recorded; therefore the first R fit is NOT accepted or called
converged. Full raw estimates/error remain in artifact 12728852170.

Normalized-Cholesky covariance has unit diagonal mathematically, while
S_ii/(sqrt(S_ii)*sqrt(S_ii)) can differ by floating roundoff. Only that identity
now uses eight machine epsilons, with full 17-digit values recorded. Runtime
regression accepts one epsilon and rejects a 1e-12 change. No parameter,
likelihood, optimizer, bound or preregistered scientific criterion changes.
Native convergence/LL metadata is now written BEFORE subsequent assertions.
The actual accepted Python 2D estimate is hash/settings checked and reused
without recalibration; the original failed evidence is copied intact. R must
run again because its original convergence metadata is unavailable. 3D has
not run yet. No production/UI capability is enabled by this repair.

## Actual initial fixed-c acceptance and expansion gate

Run 40491591755 accepted BOTH fixed-c profiles, actual native R and Python
converged; unchanged criteria all passed. 2D: maximum parameter difference
2.80947e-5, common LL difference 9.45874e-10, tangent Hessian min eigenvalue
.27943, condition449. 3D correlated within-item: parameter difference
3.63907e-5, common LL difference 1.57161e-9, min eigenvalue .090758, condition
1314. Two-step Hessian relative changes <5.7e-9; no active box bounds.
Native Python stopping is relative-objective stagnation, with projected
scores 3.31e-5 / 4.18e-4. These are numerical convergence diagnostics, not
exact stationarity or certification. Full values/source settings retained in
fixed_3PL_initial_evidence.json and the linked artifact. The R normalization
identity guard now records actual 17-digit diagonal values.

Next request stages are explicit: remaining fixed-c 3PL structures, then
free-c 3PL; independently, fixed-asymptote 4PL after fixed-c 3PL acceptance;
free 4PL only after free-c 3PL and fixed 4PL acceptance. Each stage stops its
own dependent fits on failure, while mathematically independent stages may
continue. Failed/BLOCKED stages keep the overall workflow red; no skip/PASS
substitution. Accepted initial 3PL fits and all PR80/81 fits are NOT requested
again. Each new R fit has a documented 600-second operational budget,
without changing its algorithm, bounds, tolerance or 3000-cycle limit. A
resource timeout is BLOCKED, not convergence or scientific acceptance.

## Expansion outcome and production scope (actual evidence)

Run 40493446306: all four fixed-c 3PL structure/dimension combinations and
all four fixed-c/u 4PL combinations are now independently numerically
accepted (initial cases retained, not recalibrated). Actual 2D between-item
free-c 3PL also converged in Python/R (267/1202 iterations), with maximum
parameter difference .000342845, probability .0000304860, information
.000372706 and common LL 2.73221e-8. Its reduced/tangent Hessian is positive
(min .11923, condition10703); active c=0 boundaries are reported. No regular
SE/CI/Wald or Type-I-error certification follows.

**The overall expansion run FAILED and is preserved as FAIL.** 3D within-item
free-c Python fitting converged (734 iterations, LL -3573.059780112164), but
native R did not finish within 600 seconds. R convergence is UNKNOWN, not
PASS or declared NOT CONVERGED. This is RESOURCE BLOCKED / NOT CERTIFIED;
no speculative rerun, increased budget or relaxed tolerance is requested.
The original dependent free-4PL stage did not execute. Complete mixed-success
records are in expansion_evidence.json/artifact12729839040. Model support is
based on individual actual accepted profiles, not the mixed run's label.

A separate single 2D free-4PL comparison is requested because the required 2D
free-c 3PL and fixed 4PL profiles DID pass. The request checks both exact
profiles, convergence, constraints, every numerical comparison and unchanged
criteria hash before fitting. It does not bypass the 3D prerequisite: no 3D
free-4PL fit is requested and that route remains disabled.

The byte-identical independently fitted Python engine moves to
core/irt_multidim_general.py; validation/mirt_phase3/engine.py is a compatibility
import, not a second implementation. The new experimental production adapter
exposes only accepted fixed 3PL/4PL positive CFA (2D/3D, full correlations,
between/within-item) and 2D BETWEEN-item free-c 3PL. It rejects unverified
free-c dimensions/cross-loadings before calibration. Native Python mirt
constructors remain unsupported and unchanged. Backend attribution is explicit
in UI, support matrix, method report and downloads. No local-estimator claim
is falsely attributed to Python mirt.

The existing UI/report layout is reused: true native optimizer status/trace,
physical scores/boundaries, A/delta/c/u, full latent correlations, original IDs,
EAP/conditional full posterior covariance, IIF/TIF/Fisher SEM, curves and
CSV/XLSX/PNG/ZIP. GH15 and registered solver defaults are the general-route
defaults. MAP is explicitly NOT VALIDATED/DISABLED; interactive Hessian/parameter
SE/CI/LR remain unavailable, rather than fabricated. Nonconvergence suppresses
person scores, information/curves and PNG. Any relevant data/settings change
invalidates all results/downloads. Report rows are capped at250000 BEFORE
fitting to limit Streamlit/openpyxl memory; fit budget remains20M N*J*nodes.

General 4D/5D are closed: the validated GH15 tensor would require50625/759375
nodes, above the4000-node and20M response/node operational budgets. Lowering
quadrature simply to activate them is not scientific validation. All PR80/81
native and exact-composition models remain available unchanged.
