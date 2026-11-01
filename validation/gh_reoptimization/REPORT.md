# GH101-first reoptimization feasibility — single dataset, NOT CERTIFIED

**Decision:** GH101 optimization, starting from the saved GH61 solutions, resolved the protected numerical issue for this dataset. All7 unique solutions (8 hypothesis slots) passed an **independent higher-order GH161 audit**. The fallback GH161 optimization/GH201 audit was not run because the predeclared stopping rule was satisfied. This does not establish nominal FWER, general GRM validity or a global optimum.

## Frozen protocol and unchanged inputs

Protocol **GH-REOPT-1**, commit `3788755`, was committed before reoptimization outcomes. Executable revision/source hashes are in `results/report.json`. Base is PR124 head `aee9926`, stacked on unmerged PR123. Only saved `failed_0881_minus.npz`, seed2026110981, impact−.5 was used; no generation, Monte Carlo restart or ordinal refit.

Same4-category GRM, N500/500, P01–P08 shared slope/threshold anchors, P09/P10 hypotheses, referenceN(0,1), focal mean/SD free. Native bounds, ordering, L-BFGS-B maxiter2000/ftol1e−14/gtol1e−10 and convergence/information rules unchanged. Native G.Fitter.fit was reused directly. Only the validation instance quadrature configuration was set to fit101 and **SciPy probabilists GH161** for audit, independent of native NumPy physicists fit nodes. Start clipping was checked absent; each unique hypothesis used its own archived GH61 start, not a selected alternative.

## Execution and all-hypothesis gates

**7 fits attempted/7 completed**, one per unique hypothesis; H2 shared between P09/P10. No fallback, extra starts or retries. Total **11.7733 seconds** including independent derivative/dense-information checks; native fit calls sum **1.69328 seconds**. Peak process RSS **141.332MiB**, below1024MiB; all calls remained within30s and total120s protocol limits.

| Unique slot | Iterations | Native GH101 max score | Independent GH161 max score | GH161 dominant coordinate | abs ΔlogLik GH101/GH161 | Native / dense minimum information eigenvalue |
|---|---:|---:|---:|---|---:|---|
|P09.H0|7|7.254798e-08|7.248808e-08|P10.focal.log_gap_b3|4.788188e-07|8.304815 / 8.304815|
|P09.Hb|8|7.916174e-08|7.690891e-08|P08.shared.b1|4.976606e-07|7.955279 / 7.955280|
|P09.Ha|8|6.970203e-08|6.842225e-08|P10.focal.log_gap_b3|7.249579e-07|7.939061 / 7.939061|
|P09.H2|12|5.943336e-08|5.701206e-08|P10.focal.b1|9.676917e-07|7.175931 / 7.175932|
|P10.H0|12|4.853591e-08|4.871233e-08|P10.shared.log_gap_b3|1.061897e-06|8.193300 / 8.193300|
|P10.Hb|12|4.487390e-08|4.568456e-08|focal.mean|1.060103e-06|7.794125 / 7.794126|
|P10.Ha|12|3.599983e-08|3.337417e-08|P09.focal.b1|9.960353e-07|7.852751 / 7.852751|

All7 native optimizer scores<2e−6; all higher-order scores≤1e−5 (maximum **7.69089124e-08**), no active-bound flags, exact anchor equality and ordered thresholds. Native and GH161 observed-information matrices passed the SAME relative positive-definiteness criterion min>1e−7×max(max,1); maximum dense condition **142.061400**. Native covariance availability was checked; no SE/CI/p output was generated.

For the previously worst P10.H0 solution, the saved GH61 point had GH161 logSD score−1.27874769e−5. After GH101 optimization, its GH161 maximum is **4.87123e−8** (a threshold-gap coordinate); logSD component is **-8.78026397e-10**. The objective/coordinate system was not rescaled to obtain this result.

## Independent checks

Every fitted solution received a higher-order score/logLik audit, a GH161 observed-information Hessian check using the native score and central differences, and independent direct-respondent likelihood finite differences over ALL free coordinates. Fitting and checking used different grid densities and independently constructed node/weight conventions. No solution was accepted only because its gradient was small on its fitting grid.

The preserved normalization is −logLik/N, N1000; focal transformations and shared-anchor derivative accumulation are unchanged. Two finite-difference steps are1e−4 and1e−5×max(1,abs(parameter)); their maximum analytic-score discrepancies are **2.449769e-09** and **2.306794e-10**, respectively. The arithmetic tolerance is inherited from PR124; it is not a relaxation of the scientific1e−5 gate. Max native/direct-row logLik discrepancy **3.637979e-12**. Gaussian weights/variance moments and physical constraint equality are preserved in the machine record.

## LR numerical sensitivity — no p values

All native nested-LR checks passed, with unchanged4/3/1 degrees of freedom. The table separates integration-rule changes at fixed estimates from reoptimization effects on the SAME GH161 grid.

| Item/test (df) | Original GH61 LR | Old fixed-parameter GH161 LR | New GH101-fit LR | New same-solution GH161 LR | New GH161−GH101 LR |
|---|---:|---:|---:|---:|---:|
|P09/omnibus (4)|4.484645014|4.484862485|4.484861618|4.484862596|9.777468e-07|
|P09/thresholds (3)|2.789575697|2.789659014|2.789658582|2.789659067|4.854664e-07|
|P09/slope (1)|3.236283244|3.236501073|3.236500241|3.236501181|9.400610e-07|
|P10/omnibus (4)|7.658128862|7.658046701|7.658046840|7.658046652|-1.884109e-07|
|P10/thresholds (3)|7.134279700|7.134272078|7.134272108|7.134272051|-5.669062e-08|
|P10/slope (1)|0.282756291|0.282692926|0.282693100|0.282692916|-1.848239e-07|

Max fit/audit LR discrepancy **9.77747e−7**; max reoptimization-only LR change at commonGH161 **1.10907e−7**. The inherited per-fit .001 logLik quadrature bound implies at most .004 LR difference by triangle inequality; .004 was NOT introduced as a new acceptance threshold. Small LR changes do not justify relaxing the protected score gate, and do not establish general p-value validity. **No p values were computed or appended to the old study.**

## Continuation proposal and limits

`next_protocol_PROPOSAL.json` proposes a **new uniform GH101/GH161 validation version**, not selective repair/pooling of the old fixed-GH61 experiment. First10 seed pairs should measure native cold-start runtime and all gates before separately budgeted100→500→2500 stages. Use the same data/seeds and archived ordinal outcomes. Apply one predeclared initialization and density policy uniformly; earlier GH61 final fits are not counted as GH101 observations. The current warm-start case is technical feasibility evidence, not a main-study replicate for that homogeneous procedure.

Cold-start/full-study cost cannot be inferred from this one warm-start case. Keep the predeclared9-component simultaneous inference family, paired comparisons, failure records and fixed-R rules; no success-based optional stopping or model-label promotion. No future stage was executed here. If uniform GH161 becomes necessary, define another protocol rather than changing selected seed outcomes.

## Integrity, tests and CI

All **152 prior canonical source/evidence files unchanged**, including PR123’s failed matrix, invalid classification and suppressed p values. No production code/default,1e−5 validity limit or inference gate changed. Full parameters, native and dense information diagnostics, gradient vectors, starting values, LR decompositions, timings, RSS, source hashes and versions are in `results/report.json`; stdout is `execution.txt`.

**4 distinct targeted tests passed**: protocol/constraints, native API/grid configuration without fit, LR arithmetic, then cached result/independent-gate integrity without calibration. New CI tests use only deterministic fixtures/cached evidence; existing mandatory checks are retained. No full regression was launched locally. Reproduce the opt-in experiment from repository root with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m validation.gh_reoptimization.experiment`; it refuses to overwrite saved evidence. Do not merge automatically.
