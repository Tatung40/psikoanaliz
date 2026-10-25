# First independent GRM calibration comparison — prepared, NOT EXECUTED

Reference: successful actual R1.48/Python1.2.0 no-fit run **40156690320**,
commit918ee78, artifact12584970949. Its 234 R checks passed; seeded GRM maximum
probability difference was 6.66e-16. Preflight is not scientific fit validation.
No model calibration was executed while implementing this comparison.

## Scope and identical target

Reuse the 500×8 ordinal fixture and existing preflight, including its 32-free-
coordinate R values table and 80-row constraint matrix. Mean0/variance1/D1,
4 categories, unidimensional GRM, ML/no item priors; same explicit GH61 nodes
and normal weights. Actual Python quadrature is checked before fitting; actual
R nodes/weights and latent identification are exported and checked afterwards.

Physical a∈[.1,5], b_s∈[-6,6], b_{s+1}−b_s≥1e-6 map to R native
−6a≤d_s≤6a, d_s−d_{s+1}≥a×1e-6, with d_s=−a*b_s. No approximate independent
threshold boxes replace these inequalities. Native d∈[-30,30] is only a redundant
consequence of the full constraints. The preflight-derived A matrix is used
unchanged as the R NLopt inequality Jacobian; free-coordinate names/indices
are verified at runtime, including the named-vector issue found in PR63.

Starts are the existing conventional predeclared GRM probe parameters, independent
of fitted coefficients. Both fit APIs receive those same starts in their own
parameterization. Public Python mirt1.2.0 bound/gap constants are checked without
mutation. Python uses EM, R uses EM/SLSQP; both have cap3000/numeric tol1e-8,
no acceleration and no SE computation. Their stopping definitions differ:
Python absolute logLik increment versus R all native parameter changes. These
are recorded rather than falsely called identical convergence tests.

## Execution and retained evidence

The registered manual workflow remains `r-mirt-validation.yml` / **Independent
R mirt validation**. Default `mode=preflight` performs no fits. To opt into exactly
one R and one Python GRM calibration (no PCM or other calibration):

```sh
gh workflow run r-mirt-validation.yml --repo Tatung40/psikoanaliz --ref audit/polytomous-r-strategy -f mode=grm-fit
```

UI: Actions → Independent R mirt validation → Run workflow → branch
`audit/polytomous-r-strategy`, mode `grm-fit` if the input is displayed. If the
default-branch registration does not show new inputs, use the CLI command above.
No main/bootstrap change is needed. R1.48/Python setup and caches are reused;
NLopt is required only in fit mode. Lightweight unit checks precede estimation.

The preflight must actually execute successfully first. Model/version/data/grid/
identification mismatch stops before fits. R errors stop without wasting a Python
fit. A completed but nonconverged R fit remains comparable diagnostically with the
one Python attempt; its overall result is NOT CONVERGED and CI fails. Neither
engine is retried. Local fit-start markers reject reuse of an existing attempt.
Do not re-dispatch a completed calibration merely to obtain a better result.

Artifacts always include the preflight and all available fit outputs: source/data
hashes, starts/settings/versions, raw native/transformed coefficients, category
curves, item/test information, R optimizer traces and likelihood history, shared
GH arrays, stdout/stderr, physical and native score coordinates, constraint
residuals/proximity labels, and JSON classification. PR63 comparison helpers and
R trace writer are reused from immutable commit18d1bcf; no copies are maintained.
The driver checks actual N(0,1)/no-prior results. Respondent common-grid logLik and
analytic physical marginal scores use stable log-logistic differences, without
probability clipping; small deterministic unit tests check the scores independently.

No scientific discrepancy tolerance is introduced: raw differences can be
MEASURED / NOT CERTIFIED, never scientific PASS. Positive bound/gap violations
are FAIL without silent relaxation. A 1e-6 near-bound label is diagnostic only.
Nonconvergence, invalid arrays or missing execution cannot yield a successful
calibration gate. Convergence flags do not prove stationarity/global optimality;
optimizer/flooring/gradient/stopping differences and SE/CI equivalence remain
limitations. No production estimators, PR63 evidence or inference gates change.

## Actual implementation checks

Six lightweight tests passed: stable likelihood against direct category products,
analytic scores against finite differences, extreme-tail stability, a mocked
single public GRM fit invocation, no calibration after failed preflight, and
existing no-fit/error-reporting behavior. Mock calls are not fitted evidence.
A prepare-only invocation against the successful saved preflight verified the
fixture, exact Python GH arrays, bounds and complete starts; it recorded both
fit-call counts as zero. Actual calibrations remain **NOT EXECUTED** until the
opt-in Actions run occurs. No simulations, full regression or local R install.

## Initialization class fix (failed run40164834396)

The preserved R stderr is: `pars input does not contain the appropriate classes,
which should match pars = 'values'`. R1.48 `utils.R::UpdatePrepList` checks each
column class against its native `ReturnPars` table. Reading the exported CSV
changes numeric/integer/all-NA column types (notably parnum and prior columns)
and loses `mirt_df`/itemtype attributes. It is not a model/constraint mismatch.

`restore_grm_initial_values` now calls the documented no-fit `pars='values'`
initializer for the identical data/model, verifies all immutable column values,
and assigns only the existing CSV starts and free flags into that native table.
Native column classes and attributes remain intact; constraints, starts, bounds,
quadrature and stopping rules are unchanged. No fits are added by initialization.

One R preflight regression reproduces the CSV class-validation failure and checks
restored column classes, complete attributes, starts and est flags, without any
calibration. It emits `GRM_initialization_classes.csv`. **R regression/runtime
initialization NOT EXECUTED locally**: Rscript unavailable. The next manually
approved mode=grm-fit run executes this regression before the same one-fit pair.
The original failed artifact is unchanged; no retry was launched here.
