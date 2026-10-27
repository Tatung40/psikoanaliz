# Phase 2: native API audit and exact restricted composition

Python mirt 1.2.0 rejects native 2–5D 3PL, 4PL and RSM constructors.
`capability_audit.json` records actual errors, successful **unfitted** 1D
controls, versions and source hashes. `LinearConstraint` exists as a utility,
but the public EM fitter rejects both it and ConstraintSet. Projection of
unconstrained GPCM estimates is not Andrich RSM ML. General within-item or
correlated-factor target models remain UNSUPPORTED; no class flag is patched.

A distinct, explicitly restricted route is mathematically available for 3PL/4PL:
known fixed c (and fixed u for 4PL), between-item single loadings and independent N(0,1) factors.
The joint likelihood factors exactly into independent 1D marginal likelihoods.
Public fit_mirt, fscores and native information methods perform every
psychometric calculation. The adapter only dispatches/assembles blocks;
zero cross-information follows from independence, not from an approximation.
Fixed c is assumed known, not estimated. No free/shared-estimated asymptotes,
SE/CI/p, formal LR or global-optimum certification. 3PL independent acceptance preceded activation of 4PL. Both native full
multidimensional constructors remain unsupported; these are explicit compositions. RSM shared **estimated** steps couple the
parameter optimization and cannot be obtained from independent RSM fits with different steps.
The exact single joint shared-step route below avoids that obstruction.

Predeclared acceptance targets are reused unchanged from
`validation/mirt_stage1/criteria.json`. Fixtures reuse existing 3000×12
binary data. Three unique six-item blocks, N=600, use documented seeded row
permutations to form synthetic composite respondents, not real person IDs.
Source/data hashes and row maps are retained. Two- and three-dimensional
assemblies reuse those same measured blocks, with no repeated calibration.
Four/five-dimensional factorization has no-fit numerical proofs only and
remains EXPERIMENTAL. Data independence/model fit is an assumption, not a test.

Each block uses D=1, latent mean 0/variance 1, ML/no priors, GH61 with identical
nodes and masses, positive a [.1,5], physical b [-6,6], identical starts and
fixed asymptotes. The R reference reuses matched_ml and its previously
runtime-verified coupled physical-b constraints, with a backwards-compatible
fixed_asymptotes option. Native parameter table classes/attributes persist.
A fixed-parameter R probability/information/weight preflight must pass before
any Python fit. One actual Python/R fit per unique block; no automatic retries.

Calibration reuses the existing `MIRT Stage 1 independent R comparison`
workflow on this branch. The requested family is explicitly versioned in
`requested_family.txt`; `none` makes no calibration request. This allows one
authorized PR-triggered run without a separate/duplicated workflow or dispatch
permissions. Accepted families are not requested again on later commits.
CLI: `PYTHONPATH=. python validation/mirt_phase2/factorized_comparison.py
--family 3PL --output validation-output/mirt-phase2-fit`.

The 12 accepted PR80 profiles are retained from successful run 40468300257
and evaluated with **all R item estimation flags FALSE**. No Python/R baseline
recalibration occurs. Prior original failures and independent reference evidence
are preserved. Workflow artifacts include complete Phase 2 settings, composite
data, hashes, native estimates, convergence, scores, curves and R logs.

## Actual 3PL/4PL acceptance and native API boundary

Run 40460608345: all three unique blocks converged independently in Python and
R, satisfying unchanged numerical targets. Joint logLik differences were
3.468e-9 (2D) / 3.756e-9 (3D); compact actual evidence is 3PL_evidence.json.
The 4PL request uses the same methodology with c=.2/u=.95 fixed, not estimated.
All estimated parameters are block-specific, so argmax of the sum of block
log-likelihoods separates. Merely factoring likelihood terms is insufficient
if unknown parameters (such as RSM common steps) are shared across blocks.

The generic public MultigroupModel also does not supply a verified RSM
workaround: RSM has a global threshold mask of length 3 for six items, while
MultigroupEMEstimator._optimize_item indexes every parameter mask by item
(index 3 is already out of bounds). No multigroup/RSM fit is attempted or
claimed. No independent per-factor RSM fits with different steps are passed
off as a shared-step model. The scalar structural-row joint fit below reuses the genuine supported
shared-step EM estimator instead; it does not use the broken multigroup route.

4PL first R preflight (run 40462716422) stopped before calibration on strict
physical `u == .95` equality. Original CSV records every u=.95 with estimation
flags FALSE. The corrected guard records full 17-digit values and differences,
checks unchanged fixed flags strictly, and allows only 8 machine epsilons for
physical logit/inverse-logit representations. An R runtime regression rejects
a 1e-12 perturbation. No data, asymptote target, optimizer, bound or scientific
comparison tolerance is changed; original failed evidence is retained.

Dense information export is capped at 250000 rows **before fitting** to bound
Streamlit/openpyxl memory. This operational guard is independent of scientific
criteria; factorization does not justify an unbounded dense q×q report.

Actual 4PL run 40463908384: all three R fits converged; F1/F3 passed. Python
F2 reached 1000 iterations without convergence. Its selected physical score
at two finite-difference steps was .00021254, versus R .00000293
(4PL_F2_score_diagnostic.json). This is residual optimization error in a weak
slope direction, not evidence of structural nonidentifiability. A single
public Python EM continuation uses its own saved F2 values, item ftol1e-14,
outer tol1e-12/max3000 and unchanged data, GH61, bounds and fixed asymptotes.
All three R calibrations and accepted F1/F3 values are retained without refits.
Original failed outputs remain in the source artifact and original_report.json.
The same source artifact contains the accepted 12 PR80 baseline profiles;
their successful per-profile status is verified independently of the mixed
run's overall failure. One artifact download supplies both retained baselines
and the actual converged 4PL reference, avoiding duplicate downloads.


Actual 4PL continuation run 40468300257: F2 converged after 264 additional
public Python EM iterations; all unchanged criteria passed against the retained
converged R reference. Maximum F2 a difference .0003585073, probability
.00001669887, information .0004567116, common logLik 2.95222e-9.
Joint 2D/3D logLik differences 3.65e-9 / 4.03e-9. This is numerical
reproducibility for fixed known c=.2/u=.95, not validation of estimated
asymptotes, global optimality, Type-I error or parameter CI coverage.
Compact provenance/results: 4PL_evidence.json; original failures preserved.

## Genuine globally shared-step Andrich RSM (restricted composition)

Let each item load with unit slope on exactly one independent N(0,1) factor.
For each original person, create q structural rows: axis d contains only that
axis's responses; all other cells are missing (-1 Python / NA R). Every
observed response occurs exactly once. The observed marginal likelihood is
product_n product_d integral product_(j in d) P_j(x_nj | theta_d, beta_j, tau)
phi(theta_d) dtheta_d. This equals the qD likelihood **for all common tau**.
One public RatingScaleModel / EMEstimator fits ALL structural rows jointly,
with ONE globally estimated tau vector. The step M-step pools ALL dimensions;
there are no independent per-factor RSM fits and no GPCM proxy/parameter
projection. Subject correlation across factors is excluded by the model.
A parameter covariance is NOT assumed block diagonal: estimated shared steps
couple item estimates even though conditional person information is diagonal.

Canonical gauge: tau_1=0 fixed, beta_j and remaining C-2 tau values estimated;
J+C-2 free parameters, all free physical bounds [-6,6]. Andrich steps need NOT
be ordered (GRM ordering is not imposed). Native R mirt 1.48 itemtype='rsm'
uses gpcmIRT kernel increments theta-b_k+c_j: R b_k=tau_k and c_j=-beta_j.
technical internal_constraints=FALSE disables its alternate origin constraint;
explicit CONSTRAIN groups tie b2... across ALL items, a1=1 and b1=0 fixed.
This mapping/constraints/GH61 identity must pass an actual fixed-parameter R
preflight before either engine's registered comparison fit is run.

Reproducible command (R mirt 1.48 required):
`PYTHONPATH=. python validation/mirt_phase2/factorized_comparison.py --family RSM --output validation-output/mirt-phase2-fit`.
The existing 500x8 ordinal fixture supplies disjoint row blocks with a fixed
permutation seed 20261027; 2D N200/J12 and 3D N150/J18 composite datasets,
original categories unchanged. One Python and one R joint fit per case;
no baseline/3PL/4PL refits. GH61, ML/no priors, fixed normal mean0/variance1,
matched starts/physical boxes/gauge; same unchanged preregistered criteria.
Full native values, constraints, versions, settings/hashes, logs, two-step
physical score diagnostics, probability/info/EAP/SD/logLik/AIC/BIC differences
and boundary flags are artifacts. AIC uses J+C-2; common BIC uses ORIGINAL N,
native structural-row BIC using N*q retained separately and explicitly named.

Actual RSM run 40471554619 accepted both 2D/3D comparisons; compact
measured evidence is RSM_evidence.json. No Python-only test is used to
certify R equivalence. All user analyses remain
EXPERIMENTAL / NOT CERTIFIED; q4/q5 have fixed-parameter no-fit identity tests
only. Correlated, within-item, exploratory RSM, parameter SE/CI/p and formal
LR/model-fit claims stay disabled. EAP SD / Fisher SEM are conditional on fitted
item parameters and exclude calibration uncertainty; MAP is provided by the
native Python library but not independently certified. Dense report and
structural-row computation budgets fail before fitting.

RSM first run 40470531009 passed mapping/information preflight and the Python
2D fit converged in 121 iterations (LL -3187.126636462431). R fitting then
failed the final COV_11=1 assertion. R mirt 1.48 model.elements passes
Rasch=all(itemtype in Rasch/rsm/Tutz/PC1PL) to LoadGroupPars, which sets the
variance estimation flag TRUE. This default was unsuitable for our declared
fixed N(0,1) scale. The validation now explicitly fixes MEAN_1=0/COV_11=1
BEFORE fitting, retains native group defaults and raw values even if a later
assertion fails. No observed data, bound, target scale or acceptance criterion
changed. The off-scale R attempt is NOT accepted and its original artifact is
retained (12718178692). The already converged Python 2D parameters/metadata
are hash-checked and restored through a public fixed-parameter EM
reevaluation (ALL estimation masks FALSE; bit-identical parameters, no
optimization) for native scoring; the original convergence/iterations remain
unchanged, not replaced by the reevaluation status. No recalibration; 3D had not been executed. Original failed reports are copied
into original_evidence and the source run remains accessible.


## Final measured scope

| Family | Actual independent comparison | Scope / remaining limits |
|---|---|---|
| 3PL | 2D/3D numerically accepted; 40460608345 | Known fixed c=.2, independent between-item; generic MD constructor unsupported |
| 4PL | 2D/3D numerically accepted; 40468300257 | Known fixed c=.2/u=.95; one justified own-Python F2 continuation, all R fits retained |
| Andrich RSM | 2D/3D numerically accepted; 40471554619 | ONE globally estimated shared-step vector, fixed unit single loadings; exact structural-row representation |
| PR80 12 profiles | Accepted evidence retained and reevaluated only | No Python/R recalibration; missing/rejected evidence now fails closed |
| q4/q5 restricted additions | EXPERIMENTAL / NOT CERTIFIED | Fixed-parameter no-fit likelihood/information proofs only; no R calibration claim |
| Correlated / within-item / exploratory 3PL/4PL/RSM | UNSUPPORTED | Public native constructors reject MD; exact composition requires independence/single loading |

RSM 2D: Python/R converged in 121/105 EM iterations, joint LL
-3187.126636462431 / -3187.12663646206. RSM 3D: 122/114 iterations,
LL -3593.274687719281 / -3593.27468771892. Across both, maxima:
b difference 2.14691e-6; shared-step difference 9.89783e-7; category
probability 7.26593e-7; IIF 1.22680e-6; TIF 6.58723e-6; EAP 1.33173e-6;
posterior SD 9.94792e-8; conditional SEM 2.88089e-6; common LL 3.65617e-10.
AIC/common-original-N BIC differences <7.4e-10. No active physical bound.
Two-step score diagnostics are stable: maximum Python score 1.16549e-4,
R 9.59517e-7. Absolute-LL Python stopping and parameter-change R stopping
are distinct; these scores are not new acceptance criteria or proof of exact
stationarity. No Hessian/SE/CI/inference or global-optimum claim follows.
Numerical acceptance uses UNCHANGED preregistered criteria, not observed-gap
tuning. These results do not assess population recovery, Type-I error,
coverage, fit of the fixture to RSM, or robustness to model misspecification.

Final versioned request is `none`: no accepted Phase2 family is automatically
refitted. Opt-in CLI commands remain reproducible and do not retry themselves.
Workflow source 40471554619 retains all 12 accepted PR80 profiles plus all RSM
inputs/outputs and original failure evidence. Artifact retention is 30 days;
expired/unavailable retained evidence must be restored explicitly, never
silently replaced by a new fit. Persistent compact evidence and hashes remain
committed; complete raw outputs, versions, constraints and R logs are attached
as Actions artifacts at the linked runs.

Local full regression (before the isolated RSM addition): 1120 passed,
10 existing skipped. New/affected RSM checks were targeted locally; actual
R/Python workflow focused suite: 65 passed. Final branch CI checks and
post-merge main's existing automatic full regression verify the integrated
source; no second manual full suite or old R calibration is requested.
