# Phase 2: native API audit and exact restricted composition

Python mirt 1.2.0 rejects native 2–5D 3PL, 4PL and RSM constructors.
`capability_audit.json` records actual errors, successful **unfitted** 1D
controls, versions and source hashes. `LinearConstraint` exists as a utility,
but the public EM fitter rejects both it and ConstraintSet. Projection of
unconstrained GPCM estimates is not Andrich RSM ML. General within-item or
correlated-factor target models remain UNSUPPORTED; no class flag is patched.

A distinct, explicitly restricted route is mathematically available for 3PL:
known fixed c, between-item single loadings and independent N(0,1) factors.
The joint likelihood factors exactly into independent 1D marginal likelihoods.
Public fit_mirt, fscores and native information methods perform every
psychometric calculation. The adapter only dispatches/assembles blocks;
zero cross-information follows from independence, not from an approximation.
Fixed c is assumed known, not estimated. No free/shared-estimated asymptotes,
SE/CI/p, formal LR or global-optimum certification. 4PL is not enabled before
3PL independent acceptance. General RSM shared **estimated** steps couple the
blocks and cannot be obtained from independent RSM fits with different steps.

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

The 12 accepted PR80 profiles are downloaded from successful run 40453192278
and evaluated with **all R item estimation flags FALSE**. No Python/R baseline
recalibration occurs. Prior original failures and independent reference evidence
are preserved. Workflow artifacts include complete Phase 2 settings, composite
data, hashes, native estimates, convergence, scores, curves and R logs.
