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
