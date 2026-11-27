# Advanced MIRT iteration ledger (Policy D)

Criteria: `validation/mirt_stage1/criteria.json`, unchanged; its SHA-256 is recorded in every run summary.
No thresholds were relaxed, no simulation sizes were increased, and no failed experiments were repeated.

| # | Mechanism | Hypothesis | Diagnostic | Predeclared criterion | Decision |
|---|-----------|------------|------------|-----------------------|----------|
| 0 | Polytomous bifactor | API support | `mirt.bfactor` on 3-category data | API accepts the data | UNSUPPORTED API → WITHHELD |
| 0 | M3PL/M4PL in production | — | `core.irt_multidim.fit(...,'3PL')` | — | Rejected by production (correct); phase3 evidence only |
| 0 | R backend | Deployability | Dockerfile/packages.txt audit | R mirt is present in the image | BLOCKED (shared deployment files) |
| 1 | GRM_EFA_2D probability gap, Bifactor_4D | Python EM stopped before stationarity | one fresh R+Python calibration, then the existing stage-1 precision continuation (tol 1e-11) | all stage-1 checks PASS | see `summary.json` artifact of `mirt-advanced-reference` |

Iteration-1 results are added below once CI completes; they are not edited afterwards.

## Iteration 1 result — run 45938013141 (criteria SHA 5b666c89…, unchanged)
- GRM_EFA_2D: baseline NUMERICAL DISAGREEMENT (probabilities). Precision continuation: all checks PASS. Decision: **PASS** (numerical, 2D, precision settings only). The hypothesis is supported.
- Bifactor_4D: baseline NOT CONVERGED, because both engines hit the iteration cap. The continuation precondition (converged fits) was not met. Decision: **BLOCKED**. No iteration 2 was run, because raising iteration budgets requires changing the shared stage-1 settings (Codex-owned).
- The workflow is now manual-only (workflow_dispatch), so the experiment is not repeated.

## Phase II — run 45943641718 (criteria SHA 5b666c89…, unchanged; archived as `phase2_run_45943641718_summary.json`)
| # | Mechanism | Hypothesis | Diagnostic | Decision |
|---|-----------|------------|------------|----------|
| 0 | Python graded bifactor API | `bfactor(model="graded")` / `GradedBifactorModel` exist | installed source + PyPI 1.2.0 wheel inspected | **API ABSENT**: `bfactor` has no `model` argument and rejects any response > 1 in `BifactorEMEstimator.fit` (`MirtDataError: requires dichotomous 0/1 responses`); no such class exists |
| 1 | GRM/GPCM bifactor | expressible as confirmatory orthogonal MGRM/MGPCM | `core.mirt_advanced` vs native R mirt, common GH7 grid, predeclared seeds | GPCM **PASS**; GRM FAIL (probabilities only, 4.0e-4) |
| 2 | GRM bifactor probabilities | same GRM M-step precision mechanism as GRM EFA | explicit high-precision continuation, same fixture and criteria (run 45945188758) | **PASS**: max ΔP 1.2e-6, ΔLL 9.8e-11 |
| 1 | M3PL/M4PL fixed through the R backend | backend = verified phase3 R specification | 6 archived cases | **PASS** (fidelity and independence) |
| 1 | GRM EFA 2D profile | explicit continuation passes | `fit_grm_high_precision` vs fresh R | **PASS**; ΔLL 3.3e-6 vs max ΔP 3.5e-4, so a log-likelihood trigger is unreliable → no adaptive trigger |
| 1 | Binary bifactor 4D | full-grid iteration cap | R and Python bfactor DR, quadpts 21, 1000 iterations | still at the cap in both; loadings differ by up to 1.30 → **BLOCKED** (no correctable defect) |
| 0 | Dockerfile | builds | docker job | the existing TAM check compared "4.3-25" with `as.character(packageVersion())` = "4.3.25" and could never pass; fixed to "4.3.25" |
| 1 | Docker mirt install | install_version works on bookworm | docker job (run 45945188758) | FAIL: RcppParallel needs cmake; fs needs libuv; Deriv 4.3.0 needs R >= 4.5 (bookworm ships R 4.2.2) |
| 2 | Docker mirt install | system libraries + Deriv 4.2.0 pre-pin | docker job (run 45946079172) | **PASS**: image builds; backend ready at build time; offline fit logLik matches archived direct R to about 1e-11 |
