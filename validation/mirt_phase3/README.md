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

## Free-4PL original result and one diagnostic precision continuation

Run40497502818: actual 2D free-c/u Python fit converged (489 iterations),
R reached3000 cycles and explicitly did NOT converge. Every numerical gap
nevertheless met the unchanged targets: parameters .000320519, P .0000836335,
IIF .000190737, common LL1.95082e-8; reduced Hessian min .079934, condition
16086. Outcome remains **NOT CONVERGED / NOT CERTIFIED**, not PASS. Original
actual evidence is free4_original_evidence.json/artifact12730957695.
Boundary diagnostics retain max physical score5.7153, projected/KKT score
.000406877 and transformed-logit score .0000142988. Large unprojected scores
at valid active bounds do not by themselves imply an improving feasible
step. Conversely, small likelihood gaps never override native nonconvergence.

Actual mirt1.48 hasConverged converts NAMED g/u logits using plogis, then
requires ALL absolute parameter changes <1e-8; it is not simply a raw-logit
change rule. Original EM history was not saved, so the exact failing final
coordinate/cause is UNKNOWN. A precision-floor hypothesis is consistent with
SLSQP xtol_rel1e-8 allowing a/d changes as large as5e-8 at slope bounds5,
above the outer1e-8 criterion. This is a hypothesis, not a fabricated trace.

ONE R-only diagnostic continuation now uses its OWN saved native estimates,
SLSQP xtol_rel1e-10, unchanged outer1e-8/max3000, the same600s budget,
ML/no priors/GH15/data/parameter bounds. Python's actual converged fit is
retained without recalibration. This tightens an inner solver; no scientific
target is relaxed. Existing optimizer_trace.R records full internal/LL history
and the exact last physical convergence pair against the installed rule.
Native status/estimates are saved BEFORE subsequent diagnostics.

No speculative BL alternative fit is run. An initial R-layer interpretation
that custom-group covariance was frozen is WITHDRAWN: src/Misc.cpp reloadPars
mutates the S4 parameter objects in place. Reading the original pars reference
alone does not demonstrate a frozen covariance or non-equivalence. Runtime source bodies are saved
alongside the named-asymptote/strict-threshold no-fit regression checks.
If this single precision continuation still fails native convergence, free
4PL stays disabled; no further retries, larger iteration limit or relaxed
threshold is scheduled. The3D free-c resource block also remains unchanged.

## Final accepted scope and unresolved limits

R precision run40499970808 converged after110 ADDITIONAL native EM cycles,
using only its own saved R starting estimates and tighter inner precision.
Python was NOT recalibrated. All original failed evidence remains unchanged.
Installed native last-pair rule returned TRUE, maximum last change0, below
unchanged1e-8; native convergence TRUE, not at the3000-cycle cap. Full internal
and LL histories are artifact12732009146; the exact check is retained in
free4_native_convergence.json. This does NOT prove that inner precision alone
caused the original failure, exact stationarity, or a global optimum.

2D between-item free4PL numerical gaps: parameters .000156941, P .0000144295,
IIF .000184896, TIF .000156281, EAP .0000259781, SD .0000246301,
common logLik7.45740e-9. Both active-tangent Hessians are positive and stable
at two steps (min .079934/.079956; condition16086/16082). Active c=0/u=1
bounds, conditional uncertainty and nonregular-inference limits remain.
Outcome is **NUMERICALLY VALIDATED / NOT CERTIFIED** for this measured case,
not a relabeling of the original NOT CONVERGED attempt.

| Production route | Dimensions/structure | Independent measured status |
|---|---|---|
| Fixed-c3PL | 2D/3D; full correlations; between/within | All four profiles numerically accepted |
| Fixed-c/u4PL | 2D/3D; full correlations; between/within | All four profiles numerically accepted |
| Free-c3PL | 2D between-item ONLY | Native R/Python converged; numerical targets met |
| Free-c/u4PL | 2D between-item ONLY | Native R precision/Python converged; numerical targets met |
| Free-c3PL3D | DISABLED | R resource timeout600s; native convergence unknown |
| Free4PL3D and free within-item2D | DISABLED | Prerequisites/independent calibration absent; NOT CERTIFIED |
| General4D/5D, negative-loadings, exploratory, orthogonal general backend | DISABLED | Not independently validated in this component; older supported routes remain |
| MAP / parameter SE/CI / LR/p / population certification | DISABLED | Independent scientific evidence absent |

Every user analysis remains EXPERIMENTAL / NOT CERTIFIED. The registry
eligibility regression checks EACH enabled family/dimension/structure/free-
asymptote tuple against actual converged R/Python results, every unchanged
comparison and criteria hash; original failed rows and the3D block cannot
activate a route. No validation is inferred from a model name or CI alone.

Final request is EMPTY/NOT REQUESTED. No accepted Phase1/2/3 profile is
silently recalibrated on future CI. Existing R setup/cache/workflow is reused;
the retained12 PR80 profiles are only reevaluated at saved estimates. Full
raw diagnostics/artifacts have30-day retention; complete original failed
outputs stay linked and compact persistent evidence/versions/hashes are
committed. Unavailable evidence fails closed, not replaced by fabricated data.

Validation of final integration: one manual full regression runs on the
stable engine/three initial UI routes in the isolated worktree. The newly
accepted fourth (free4PL) UI route and registry eligibility are checked with
focused tests without restarting that suite. Existing main push CI runs the
complete final source after merge. No second manual full regression or new
Monte Carlo/accepted-model calibration is requested.


Correction audit: runtime_contract.R (NO IRT FITS) checks the actually resolved
mirt1.48 optim namespace and its bounded BL dispatch using only a scalar
quadratic API probe. EM.group selects method="nlminb" for finite bounds;
compatibility must be established in the installed namespace, not guessed
from an R-only code slice. This does not certify BL calibration or replace any
actual EM evidence. Exact runtime diagnostics are workflow artifacts.
