# 3PL/4PL independent person holdout: amendment F

Preregistered commit: 72d1d651aa2433fc416cbd960dc7671da9c31e2f; PR execution head: 8059811660688008ce483908c67baefad48dd5e1; actual GITHUB_SHA (synthetic PR merge): 44639756faffe1c3baa799a69acc7a29ee6454c9.
Base main 7fc9161 includes PR #187. Actual run: https://github.com/Tatung40/psikoanaliz/actions/runs/45593803451.

## Actual execution
Exactly one native MAP calibration per model (N=3000, J=12), both converged; no retries. Existing default priors, GH61, tol 1e-10, max_iter 50000.
A separate seeded N=10000 holdout retains independent true N(0,1) theta and uniform draws; responses use the ORIGINAL TRUE item parameters, not the fitted parameters.
Production GH121/GH241 scoring is reused without modification.

| Model | Coverage of diagnostic EAP ± 1.96 posterior SD | Conditional MCSE | 97.5% exact CP | Bias | RMSE | r | Fit / total seconds |
|---|---|---|---|---|---|---|---|
| 3PL | 9522/10000 = .9522 | .00213343 | [.94719834, .95686396] | -.01603797 | .58876869 | .80635569 | 3.65 / 4.14 |
| 4PL | 9499/10000 = .9499 | .00218151 | [.94479154, .95467258] | .00583606 | .62781720 | .77690851 | 5.70 / 6.11 |

Maximum GH121/GH241 discrepancies: 3PL 1.6542323e-14; 4PL 2.7089442e-14.
Each isolated runner passed the same six lightweight bookkeeping tests before its fit (6 tests, not 12 distinct tests).
Artifacts person-holdout-3PL/person-holdout-4PL preserve calibration.npz, holdout.npz, original theta, response matrices, estimated_items.csv, holdout_scores.csv, report.json and complete logs. Exact hashes, versions, settings and theta-band results are in the archived JSON.
The entire pilot is new evidence, not a rerun of the failed item-interval campaigns.

## Interpretation
**MEASURED / NOT CERTIFIED.** Both marginal coverage points pass the unchanged .95 ± .02 diagnostic screen.
The CP intervals and MCSE describe independent holdout persons CONDITIONAL ON ONE calibration. Bonferroni gives at least 95% simultaneous coverage for these TWO fixed-calibration proportions only.
Ten thousand people are not ten thousand independent calibration repetitions. This does not certify unconditional repeated-calibration coverage, uniform theta-conditional coverage, J30 or a full scope rectangle.
The mean ± 1.96 SD interval is a normal approximation, not an equal-tailed posterior interval. No new person CI is exposed in production.
No prior person bias/RMSE threshold exists; those statistics are descriptive, not newly invented acceptance tests.

## Remaining parameter-interval and item-fit blockers
The preserved A2 evidence reports 3PL N3000/J12 b/c coverage .976111/.974993. The preserved A1 sampling-covariance attempt has 69/300 refused 3PL fits and 218/300 refused 4PL fits in that cell; surviving 3PL b/c coverages .922439/.897908 do not repair the gate.
A2 4PL posterior-curvature a coverage reaches .998333. An inverse penalized Hessian is not the penalized estimator's repeated-sampling covariance; simply substituting the already-failed sandwich, rescaling SEs against observed rates, or relabeling transformed endpoints is not a valid repair.
The archived A1/A2 files contain aggregate replicate coverage records, not the full joint fitted parameters/Hessians needed to validate a genuinely new covariance method retrospectively.
A bias-aware interval or regular ML profile method requires a new preregistered independent multi-calibration study, including boundary and weak-identification cases; no naive chi-square pivot is asserted for MAP or boundaries.
For S-X², a frozen-fit response simulation omits refitting and grouping/estimation effects. Numerical statistic identity is insufficient for p calibration. A validated refit-aware null calibration needs a separate study; no uncalibrated p-value is opened.

## Production decision
All previously certified item estimates/IRF/information, posterior scoring, convergence/boundary/covariance safeguards and PR #187 UI/XLSX/ZIP contracts remain unchanged.
No certification registry cell is promoted; parameter CI and S-X² p-value gates remain closed as before.
No R execution or full local regression occurred. Cloud execution is unavailable; actual Python execution is evidenced by the linked Actions job.
Zero remaining scientific blockers is NOT established.

## Operational replay and provenance correction
GitHub PR path filters use the whole PR diff. The archive-only commit therefore automatically replayed the identical seeded calibrations in run 45594344360; those are not new independent calibration repetitions and are not pooled with the canonical run. Its separate artifacts remain preserved. The workflow now verifies exact scientific-source bytes against the original head and validates the canonical JSON before reusing it, preventing further fits on documentation/archive pushes.
The initial archive test incorrectly equated PR head SHA with pull_request GITHUB_SHA. The actual runtime uses synthetic merge 44639756faffe1c3baa799a69acc7a29ee6454c9; the original workflow head is 8059811660688008ce483908c67baefad48dd5e1. The test now checks the exact recorded runtime SHA. No numerical result, protocol, bound or acceptance criterion was modified.
