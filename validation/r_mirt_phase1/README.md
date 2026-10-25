# Independent R mirt validation — Phase 1

Base: `db16a5a` (PR #62 merged). Application estimators and inference gates are unchanged.
**Current execution: PENDING.** This environment has no `Rscript`. No R fit was executed,
no package was installed, and no model is independently validated by this phase.
`pending_report.json` records the actual environment, fixture checksums and skipped cases.
Historical R fixtures in `tests/fixtures/{rasch,irt_b10,irt_b11}` remain historical evidence;
they are not relabeled as executions of the current Python mirt application adapter.

## Execute only when R is available

From the repository root, using its installed Python environment:

```sh
python validation/r_mirt_phase1/harness.py --output /tmp/r-mirt-comparison --models 1PL 2PL
```

This performs at most two single-group comparisons, not Monte Carlo studies. R must
already have `mirt` installed; package absence is recorded PENDING, never auto-installed.
Other models require an explicit `--models` list. To prepare all case manifests in an
environment without R, use `--models 1PL Rasch 2PL 3PL 4PL GRM PCM GPCM NRM RSM`.
Run the R side alone with `Rscript validation/r_mirt_phase1/reference.R MODEL data.csv OUTDIR`.
R produces category traces, item information, common parameter coordinates, raw parameters/
covariance, diagnostics, package/R version and `sessionInfo.txt`. Python never supplies R
with estimated coefficients. Fresh R output directories prevent reuse of stale output.
Do not add these fits to ordinary CI; the unit tests mock the subprocess/file protocol.

## Model equivalence contract

Common logistic constant **D=1**; no 1.702 conversion. Unchanged item-only committed CSV
fixtures are reused; their paths, shapes and SHA256 are in the report. The nominal browser
fixture uses labels A/B/C/D: the declared map A→0, B→1, C→2, D→3 is an identity convention,
**not a success order**. No imputation, row deletion, sorting of steps or category merging.

| Application case | R mirt specification prepared | Identification / common parameters |
|---|---|---|
| 1PL | `2PL`, all a1 fixed 1 | Mean0/variance1. b=−d/a; c=0,u=1. Not free-variance Rasch. |
| Rasch | `Rasch` | a1=1, mean0, variance free; compares `core.rasch`, including latent variance. |
| 2PL | `2PL` | Mean0/variance1; common a,b. |
| 3PL | `3PL`, ML only | Mean0/variance1; a,b,c (R g). MAP default application path is separately NOT COMPARABLE. |
| 4PL | `4PL`, ML only | Same plus upper d (R u); boundary/weak-identification risks remain. |
| GRM | `graded` | Mean0/variance1; physical b=−d/a from cumulative logits, not raw R intercept equality. |
| PCM | `gpcm`, all a1 fixed1 | Mean0/variance1, matching the application's restricted PCM. R `Rasch` on ordinal data frees variance and is a different model. |
| GPCM | `gpcm` | Mean0/variance1; physical adjacent step b, a; steps are not sorted. |
| NRM | `nominal` | Mean0/variance1; compare effective A=a1*ak and D relative to category0; orient theta by the first item's category1/category0 contrast. R ak and Python A are not directly equated. |
| RSM | `rsm`, a1 fixed1 | Common rating steps, mean0/variance free; compares the separate native `core.irt_rsm` default chain, not the restricted two-group DIF engine. Common adjacent steps remove origin-convention differences. |

R specification source (reviewed 24 October 2026):
https://philchalmers.github.io/mirt/reference/mirt.html — especially `Rasch`,
`gpcm/nominal`, `gpcmIRT and rsm`. These specifications are **prepared, not runtime-verified**.
Their constraints/identification must be checked against the installed R version on first execution.

## Numerical outputs and statuses

The active public Python adapter is used with ML for binary/polytomous cases. It uses
Python mirt 1.2.0 defaults (EM,21 nodes,500 iterations,tol1e-4); the exact public signature
is recorded. Native Rasch/RSM use 61-point GH. R uses EM,61 nodes,tol1e-8,3000 cycles.
Differences in quadrature/optimization are deliberately recorded, not assumed equivalent.
R/Python parameter bounds also require audit (physical b bounds do not translate to
independent R d bounds). Boundary-dependent maxima cannot certify estimator equivalence.
No estimator or application setting was changed to improve agreement.

Curves align by item/node/category; theta discrepancies are separately reported to avoid
floating-point merge failures. TIF sums per-item information once per node. Parameter and
diagnostic comparisons report absolute discrepancies and symmetric relative discrepancies:
abs(x−y)/max(abs(x),abs(y)), with zero/zero defined as zero. R g/u remain distinct from
slope-intercept d; binary asymptotes are never guessed from grid endpoints.

- **PENDING execution / SKIPPED comparison:** no Rscript or no R mirt; Python calibration skipped.
- **FAIL:** process/schema/nonfinite/convergence or trace-key failure. Failures/logs are retained.
- **NOT COMPARABLE:** mismatched parameter coordinates or unmatched MAP priors/parameter densities.
- **SKIPPED with numerical metrics:** actual executions available but acceptance tolerances uncalibrated.
- **PASS:** discrepancy helper supports an explicitly supplied tolerance, but the Phase 1 runner
  deliberately supplies none. No automatic scientific PASS is possible with current policy.

SE/CI comparisons remain **NOT COMPARABLE** until full covariance transformations,
identification and estimator/prior equivalence are audited. R raw covariance is retained;
likelihood SEs must not be treated as MAP posterior uncertainty. R/Python MAP requires
matching prior densities and transformation Jacobians, not merely prior names.
No numerical acceptance cutoff is invented from observed discrepancies. Existing native
R fixture tolerances are not silently transferred to the current Python library adapter.

## Phase 1 evidence and next step

Actually compared with new independent R execution: **none**.
Prepared/pending: 1PL, Rasch, 2PL, 3PL, 4PL, GRM, PCM, GPCM, NRM, RSM.
MAP and transformed SE/CI: not comparable yet. Targeted harness tests run without R or
simulations. A full application regression is unnecessary: no existing app/core files changed.
Next: use an already provisioned R+mirt runtime; execute 1PL/2PL, inspect version/session,
parameter constraints, quadrature and maxima, then calibrate justified acceptance criteria
before extending comparisons. Pin the actual R environment after that first real run.

## Manual binary R execution

`.github/workflows/r-mirt-validation.yml` is manual-only (`workflow_dispatch`). It
provisions official CRAN R `mirt` using `r-lib/actions` with dependency caching,
then runs Rasch, 1PL, 2PL, 3PL and 4PL once each through this harness (ML).
Python dependencies are cached too. Raw R outputs, software/settings metadata,
Python estimates/curves, discrepancies and execution logs are retained in the
`independent-r-mirt-<commit>` artifact, including failed runs. No Monte Carlo runs.
Workflow success means execution/convergence checks succeeded, **not scientific
PASS**: tolerances, optimizer/quadrature/bound equivalence and SE/CI remain subject
to the contract above. The committed `pending_report.json` is historical preparation,
not the result of this workflow. Consult the artifact tied to the actual commit.

```sh
gh workflow run r-mirt-validation.yml --repo Tatung40/psikoanaliz --ref validation-r-mirt-phase1
```

If dispatch is unavailable (credentials or default-branch workflow discovery),
execution remains PENDING; do not relabel any model as independently validated.
