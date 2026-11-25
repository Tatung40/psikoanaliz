# MAP parameter uncertainty: centered marginal-score confirmation G

Protocol commit 7c6dba3ccfa9475260053938e997f0c5d57e6078 precedes actual execution head 2211429dc7c8226f11f95a9305bb93ac57b0abd8.
Actual run: https://github.com/Tatung40/psikoanaliz/actions/runs/45606844841.
Base da0ae7c; native estimators, original prior studies and all production gates unchanged. No person or item-fit work.

## Method and root cause
The default MAP covariance is inverse penalized Hessian V, i.e. posterior curvature. It is not generally the repeated-sampling covariance of a biased MAP estimator.
A1's V(V^-1-P)V formula and analytic default-prior curvature are not repaired by repeating them. Observed likelihood curvature need not be a positive sampling-score covariance at a penalized finite-sample solution.
This genuinely distinct candidate uses V B V', where B=N/(N-1) sum of centered individual marginal likelihood score outer products. Theta is integrated using GH61; complete-data scores and prior scores are not substituted.
Native physical a/b/c/d labels determine the full joint ordering; cross-item and cross-parameter terms are retained. No transform is required because this is already the physical coordinate system. Central differences at two step sizes independently verify every analytic score coordinate before fitting.
No sign, parameter-order or missing-Jacobian defect was demonstrated in the existing production covariance. This is a different frequentist approximation, not an automatic bug fix.

## Actual independent confirmation
Eight cells ran in parallel; 1370 independent calibration datasets attempted, 1370 valid. Seeds and raw decisions were preserved; no failed seed was replaced.
Columns list a/b/c, and additionally d for 4PL. Cluster MCSE uses independent calibration datasets, not pooled item intervals.

|Model|N/J|Actual/planned R|Coverage a/b/c[/d]|Cluster MCSE a/b/c[/d]|Recorded status|
|---|---|---|---|---|---|
|3PL|3000/12|300/300|0.912778 / 0.928611 / 0.908611|0.004766 / 0.004246 / 0.005117|INCONCLUSIVE|
|3PL|3000/30|300/300|0.916778 / 0.926222 / 0.906889|0.003166 / 0.002903 / 0.003042|FAILED CRITERIA|
|3PL|10000/12|150/150|0.936111 / 0.928889 / 0.925000|0.005973 / 0.006024 / 0.006677|INCONCLUSIVE|
|3PL|10000/30|150/150|0.941333 / 0.935778 / 0.926444|0.003819 / 0.004052 / 0.004187|FAILED CRITERIA|
|4PL|3000/12|300/300|0.943056 / 0.917500 / 0.906944 / 0.764167|0.003845 / 0.004425 / 0.004811 / 0.004575|FAILED CRITERIA|
|4PL|3000/30|10/300|0.956667 / 0.933333 / 0.936667 / 0.910000|0.017249 / 0.009938 / 0.016063 / 0.013194|INCONCLUSIVE|
|4PL|10000/12|150/150|0.950000 / 0.931111 / 0.926667 / 0.857222|0.005288 / 0.006194 / 0.005947 / 0.006224|INCONCLUSIVE|
|4PL|10000/30|10/150|0.970000 / 0.916667 / 0.920000 / 0.886667|0.010482 / 0.014272 / 0.020608 / 0.021773|INCONCLUSIVE|

Two expensive 4PL J30 cells stopped after the preregistered ten-replication runtime feasibility check. They remain INCONCLUSIVE; no results were invented for unexecuted replications.
Other incomplete precision gates are explicit in each original JSON. A cell labeled INCONCLUSIVE may ALSO have failed coverage checks: all component checks are archived, and no coverage failure is hidden by that label.
All covariances/seed hashes/point estimates/errors and refusal logs are in the original per-cell artifacts; canonical summaries, artifact digests and independently derived per-item MCSE are committed. Original runtime GITHUB_SHA is retained as recorded, not replaced by the branch SHA.

## Scientific decision: NOT CERTIFIED for the new method
At N3000/J12 (300 valid independent fits), 3PL a/b/c cover .912778/.928611/.908611; 4PL a/b/c/d cover .943056/.917500/.906944/.764167.
At N3000/J30 (300 independent fits), 3PL b/c cover .926222/.906889, MCSE .002903/.003042.
The unchanged criterion .95 ± .02 is not met across the scope. Some other cells also miss the registered MCSE <=.005 requirement. Planned replication counts were not treated as a precision guarantee.
4PL d at N3000/J12 has mean SE .025744 vs empirical itemwise sampling SD .026780, yet coverage .764167. Empirical RMS item mean error is .024554. Thus closer sampling-variance estimation alone does not establish centered-normal interval validity; bias, asymmetry and heterogeneous item behavior remain relevant. This does not identify a unique causal mechanism or license a numerical SE rescaling.

A Gaussian penalized-estimation identity separately demonstrates why a plug-in bias shift cannot inherit the old SE: its Jacobian changes sampling covariance, and residual bias may remain. A bias-aware or profile procedure needs its own preregistration, uncertainty derivation, boundary treatment and fresh confirmation. No naive MAP/boundary chi-square pivot or truth-specific correction is certified.
A failed candidate is a scientific result; successful execution/integrity CI is NOT a scientific PASS.

## Production and next action
No new SE/CI method is integrated and no certification registry cell is promoted. Existing certified 3PL a, all certified point/IRF/information outputs, and PR187 UI/CSV/XLSX/ZIP contracts remain unchanged. 3PL b/c and 4PL inference gates stay closed as before.
The next defensible study must evaluate a mathematically specified bias-aware interval or valid profile pivot on new seeds; repeating this covariance approximation unchanged is not justified. Such a study cannot be inferred from the present failed confirmation.
No R execution, full local suite, threshold relaxation or post-hoc seed selection occurred. The wall-clock cutoff ends development; only existing CI observation and a conditional exact-head squash merge may follow.
