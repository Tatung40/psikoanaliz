# PR131 — final scoped closure

1PL Lord now uses the existing native Rasch calibration with a=1,mean=0 and
independent free SDs; full nuisance covariance and anchor Jacobians retained.
Actual R mirt1.48 reference passes the unchanged chi-square and common-grid
score criteria. Legacy fixed-SD inference remains closed. 2PL Lord/Wald and
original reference evidence are unchanged. See lord_rasch_free_scale/REPORT.md.

GRM-FWER was explicitly stopped by the user at852 complete pairs/1704 recorded
valid analyses. The next dispatched pair has uncollected UNKNOWN outputs.
1000/1500 looks were not reached. Only look500 supplies registered simultaneous
intervals: minus29/500=.058,[.033928,.091237]; zero25/500=.050,[.027860,.081499].
**INCONCLUSIVE / NOT CERTIFIED.** No protocol/boundary change or label promotion.

## Regression evidence

One full run:1295 passed,12 failed,15 skipped. During that run, source edits
triggered the existing project_modules mtime guard, reimporting core modules
while tests held pre-reload Recorder imports. Empty code captures, rather than
failed psychometric estimates, caused all12 failures. An isolated read-only
mtime-record counterexample reproduced the distinct ContextVars and empty
capture. With sources frozen, exactly those12 tests passed in a clean process
(237.38s). No estimator changes, assertion weakening or second full run.
The original full-run log and focused-retest log remain in the execution
workspace; the result is not misrepresented as a clean first full run.

Additional targeted runs:37 initial Lord/Wald/covariance cases;32 UI/evidence
cases;23 final reference-gate/legacy-policy cases; the updated sigma/export
case; four sequential-statistics/data cases and the archive-projection case.
These are overlapping software checks, not independent scientific replications.
The final CI includes the affected MIRT UI cases and frozen-evidence checks.
No Monte Carlo or new R fits run in routine final CI: cached actual reference
identity is verified, then comparisons/arithmetic are replayed without fitting.

Independent raw audit1704 analyses passed: saved LR/df, Holm/BH, MCSE/CP,
anchors/bounds/stored derivatives/information, source/old-record identity. No
response regeneration or fitting in the final audit. Canonical records preserve
all primary estimates/results; omitted dense matrices are reproducible caches
with saved digests. Original full archive is retained unchanged in workspace.

Remaining limits: fixture-specific Rasch numerical comparison, conditional
asymptotic inference and invariant anchors; no general Lord/Wald/SIBTEST/Raju
FWER certification. GRM scoped FWER remains unresolved. Portable replay without
the source-identified reference assets deliberately withholds 1PL Lord p/q.
No merge, main push or deployment is authorized by this finalization.
