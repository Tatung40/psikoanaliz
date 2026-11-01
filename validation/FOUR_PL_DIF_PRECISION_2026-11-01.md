# 4PL + DIF/FDR: saved-evidence precision audit

Base: main `1429945f8d0f361b89eb2cff19fef2c5030dc953` (merged #116).
No new calibration, Monte Carlo study, Hessian or independent R execution.
Old p1/p2/p6/p7 evidence is unchanged. Rscript unavailable: **NOT EXECUTED**.

## A/B — 4PL stopping decisions

- **ML: NOT VERIFIED.** p1: 880 total fits across models/conditions; every studied
  4PL ML cell failed recovery. More iterations/true starts do not establish a global
  optimum; a likelihood above generating truth does not establish convergence.
  Root-cause report wording corrected; numerical evidence unchanged.
- Saved R diagnostic: 3000 EM cycles, NOT CONVERGED; 29/50 coordinates failed the
  1e-8 native-change criterion, max change 2.1645e-6. Common-GH61 R−Python LL
  +.000228813; max ICC difference .008999804. No refit performed in this audit.
- Saved active-face Hessian: min eigenvalue .037891, scaled condition 2349.5,
  weakest physical direction 99.14% M12. Stable derivatives support practical weak
  identification, not structural nonidentifiability or certified stationarity.
  This is a stopping condition, not justification for more blind optimization.
- **MAP: CONDITIONALLY VERIFIED for point recovery only**, normal θ, library default
  priors, directly studied N=3000/10000 and J=12/30 (20 repeats/cell). Existing
  N>=3000/J>=12 policy must not be read as direct evidence for every intermediate
  design. p7 N3000/J30 has 200 repeats: coverage .971–.994 is not a valid general
  Wald-CI certificate; all 4PL MAP CIs remain withheld. Prior sensitivity and an
  independent matched R MAP comparison remain open. Do not transfer ML comparisons.
- p1's uniform [-6,6] 61-node diagnostic is distinct from GH61 fitting and the
  saved Phase1 common-GH61 comparisons. Historical p6/p7 clustered MCSE remains
  UNKNOWN: pooled aggregates cannot recover lost replication information.
- Existing 20/200-repeat evidence already answers the studied ordinary designs.
  No duplicate pilot was justified; new challenging profiles and N500 remain open.
  Packages A/B stop on weak identification/reference and precision limitations.

## C — confirmed precision defect and minimal correction

p2 pooled correlated item indicators when computing binomial intervals for Type-I
rates, power and parameter coverage. Item outcomes from a common fitted dataset
are not independent Bernoulli trials. Existing mean-FDP FDR MCSE was already based
on independent datasets and is preserved. FWER is a Bernoulli event per dataset,
so its exact Clopper–Pearson intervals are also preserved.

New Type-I/power/coverage MCSE reuses #116's cluster ratio formula. For capacity k,
C covered/rejected and M eligible per dataset, C/k and M/k lie in [0,1]. With R
independent successful datasets, ε=sqrt(log(4/.05)/(2R)), apply two two-sided
Hoeffding bounds and the union bound; the interval for E(C)/E(M) is
[max(0,(mean(C/k)−ε)/(mean(M/k)+ε)),
 min(1,(mean(C/k)+ε)/(mean(M/k)−ε))], with upper=1 if the denominator bound <=0.
These bounds are conservative and marginal, not simultaneous over the 28 cells.
Mean-FDP intervals instead use ε=sqrt(log(2/.05)/(2R)). Zero observed variance
never produces an assertion of exact zero risk. No new scientific cutoff is used.

Metrics remain conditional on successful analyses. Failed analyses retain unknown
FDP in [0,1]; the reported best/worst empirical FDR range is a sensitivity range,
not a confidence interval. Untestable items remain counted as nonrejections and
are disclosed; they cannot be treated as evidence of inferential validity.
Historical `meets_*` flags compare observed point estimates to unchanged cutoffs;
they are not precision-adjusted acceptance or permission to expose new inference.

## Reanalysis of saved records (actual execution, no fits)

2,560 records / 28 cells; one failed IRT-LR purification retained. All old point
estimates/flags are identical (targeted test); only uncertainty reporting changes.
Ranges below span existing null/alternative, impact and group-size designs:

| Method | BH-FDR | Holm-FWER | FDR MCSE | Clustered raw Type-I MCSE |
|---|---:|---:|---:|---:|
| Binary MH | .010–.060 | .010–.060 | .0090–.0239 | .0042–.0056 |
| Binary LR | .0228–.070 | .030–.090 | .0082–.0256 | .0043–.0062 |
| 2PL IRT-LR | .000–.050 | .000–.050 | .000–.0349 | .0071–.0098 |
| Ordinal Mantel | .0533–.105 | .040–.120 | .0150–.0288 | .0066–.0087 |
| Ordinal LR | .045–.110 | .020–.110 | .0132–.0314 | .0063–.0103 |

These measurements do not establish general FDR/FWER control, even in a cell with
zero empirical FDR. The conservative intervals show why mean-only checks are
insufficient. Source hashes, exact per-cell intervals/counts, failure ranges and
analysis software/code hashes are in `evidence/dif_precision_reanalysis/dif_validation_study.json`.
Original RNG seeds remain in the immutable source records; source engine versions
were not recorded there and remain UNKNOWN. Reanalysis versions are separate.

Reproduce (no estimator imports/calls in this path):

```
python -m validation.dif_validation_study /tmp/dif-precision --reuse validation/evidence/p2_dif/dif_validation_study.json
python -m pytest -q tests/test_dif_precision_reuse.py
```

## Limits and next steps

No p/q, Wald, CI or model status was newly enabled. Existing 3PL/4PL bootstrap
UI/CSV suppression, Python multigroup DIF restrictions and ordinal-impact warnings
remain. The p2 purified designs do not provide a paired fixed-vs-automatic anchor
comparison or selection-adjusted general inference. Broader dependency, small-group,
non-normal, nominal-category and bootstrap designs remain NOT VERIFIED.

First use the retained empirical FDP variance and desired Monte Carlo precision
to preregister a bounded paired anchor pilot; reuse seeds across anchor strategies
and analyze their paired differences by dataset. 20–30 repeats would be diagnostic,
not certification. No expansion is justified before choosing that precision target.

Five no-fit new tests cover perfect within-dataset dependence, zero events,
variable eligibility, failures, exact FWER, unchanged point estimates and source
integrity. Eight existing evidence/bootstrap-CSV gates also passed. A single full
regression is requested in this PR's existing CI job instead of the targeted list;
no separate full local run or redundant study is launched. Final head/CI outcome
is recorded in the PR. No production engine, thresholds, labels or main merge changed.
