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
