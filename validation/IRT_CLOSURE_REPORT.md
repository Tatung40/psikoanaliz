# Independent 3PL/4PL closure — scoped P2 certification

Protocol c7327ac8cfbd1fa7c3956a8861b202be0bc5a34a; actual native workload 45617556122; corrected read-only aggregation 45637837615.
PR #189 and #190 were reconciled only after all their CI checks passed; their failed scientific decisions remain unchanged.

## P2 independent confirmation
Each model: 300 independent N3000/J12 MAP/GH61 calibrations, common.truth item design, N(0,1); all4096 independent-person response patterns evaluated with converged posterior CDF integration and unchanged native GH241 EAP/SD.
Three separately scoped outputs pass for EACH model: population EAP recovery, equal-tailed posterior interval MARGINAL coverage, and exact raw-score inversion (fitted-item coverage on the21-point theta grid and normal-prior tail averages).

|Model|EAP bias|RMSE|Posterior marginal coverage (cluster MCSE)|Score-CI marginal coverage (cluster MCSE)|Minimum simultaneous lower grid coverage|
|---|---|---|---|---|---|
|3PL|-.0003552|.5876398|.9494500 (.0001580)|.9781047 (.00002394)|.9652394|
|4PL|-.0008499|.6271605|.9493249 (.0001511)|.9773687 (.00001899)|.9598919|

Posterior credible tail coverages are only .6500/.7852 (3PL) and .6176/.7496 (4PL). They MUST NOT be labeled frequentist fixed-theta CIs.
Exact raw-score inversion allows infinite limits or empty sets; those are genuine finite-test/asymptote limitations. Fitted-item evidence does not prove all-real-theta or J30 coverage.
Calibration independence matters: these are held-out-person results, not in-sample coverage. Production therefore retains the old calibration-person screen/status and adds a distinct independent scoring flow requiring explicit design declaration and disjoint unique IDs.
CSV/XLSX/ZIP and reproducibility assets carry interval type, scope, numerical checks and limitations. Original MAP estimates, posterior SD and parameter inference gates are unchanged.

## P1 not certified
The genuinely new full-native-bootstrap score-sensitivity binding inversion was developed on20 datasets/model, B399. 3PL b/c coverage .925/.9042 (cluster MCSE .01588/.01739); 4PL only12/20 eligible, retained intervals .9931–1.0. No pilot is certified and no independent P1 confirmation is claimed.
Gaussian Jacobian identities pass; this does not ensure stable nonlinear inversion on weak MAP ridges.
Warm full-refit benchmarks additionally demonstrate flat geometry: 4PL b can move .056 while log-posterior changes only2.43e-5, and S-X2 moves .207. Both warm candidates fail unchanged numerical equivalence gates and cannot replace canonical cold fits.
A valid full-cold independent bootstrap confirmation cannot fit the remaining wall-clock budget. No arbitrary SE rescaling, posterior/CI relabeling or pseudoinverse was used.

## P3 not certified
3000 independent refitted null datasets/model: pooled raw chi-square Type-I .0634444 (cluster MCSE .00128074) for3PL and .0855000 (.00146009) for4PL.
Per-item worst rates .076/.095; largest simultaneous upper CP bounds .090865/.111299 exceed the preregistered .07 gate. All fits/tests were evaluable.
Full fitted-null bootstrap remains a defensible alternative, but independently validated cold refits exceed remaining budget; warm/SQUAREM approximations failed the fixed S-X2 reference bound .0022. No p-value gate is opened.

## Preservation and scope
Old certificates and original binary registry bytes remain intact. New supplementary registry evidence/irt_closure/P2_CERTIFICATION.json registers only the six P2 output-method/model cells above.
No J30, alternative item design, missingness or nonnormal-calibration generalization. Bayesian credible intervals remain distinct from frequentist sets, and all intervals condition on fitted item parameters.
Raw fitted coefficients/covariances, seeds, hashes, native statistics, bootstrap draws/errors and checkpoints remain in immutable workflow artifacts. REPORT.json archives the actual compact aggregate; CI green means execution/integrity, not new certification.
