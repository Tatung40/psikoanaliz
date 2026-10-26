# Independent R mirt validation — final Phase 1 summary

**Existing evidence only; no new fits, simulations or Hessian computations.**
Machine-readable exact values, source-report hashes and fixture hashes are in
[final_summary.json](final_summary.json). No model receives a scientific PASS:
acceptance tolerances, SE/CI coverage and generalization are not calibrated here.

**4PL: PARTIAL NUMERICAL AGREEMENT / NOT CONVERGED / NOT CERTIFIED.**

Raw `SKIPPED` comparison metrics mean that no acceptance tolerance was applied,
not that R was unexecuted. The 4PL nonconvergence `FAIL` remains in its original
report; this summary does not convert it to PASS.

## Executed numerical comparisons

Absolute maximum differences across item parameters, ICC/category probabilities
and test information on the saved 81-point [-4,4] response grid:

| Model | N × items | Parameters | ICC | Test information | Absolute logLik difference |
|---|---:|---:|---:|---:|---:|
| Rasch | 1000 × 10 | 1.275656723e-6 | 3.188955857e-7 | 5.554912297e-7 | 1.352327672e-8 |
| 1PL | 1000 × 10 | .000441676754 | .000110384616 | .000266689028 | .000051490667 |
| 2PL | 1200 × 10 | .000922610776 | .000336662926 | .001333776267 | .000012731337 |
| 3PL, matched target | 3000 × 12 | .000235306457 | .000109140867 | .000094392471 | 1.537009666e-7 |
| 4PL, R warm-start diagnostic | 3000 × 12 | .064621784106 | .008999803978 | 0.012541054168 | .000228812638 |


Rasch, 1PL, 2PL and matched-target 3PL fits reported convergence in both engines;
these are **observed numerical agreement**, not acceptance-test certifications.
Rasch compares native `core.rasch` with R `Rasch` (slope1, mean0, variance free).
1PL fixes slope1 and variance1; it must not be conflated with free-variance Rasch.
2PL/3PL use mean0/variance1. All use D=1, unchanged deterministic fixtures and ML.
R intercept d maps to physical b=-d/a; R g/u map to Python c/upper d.

Rasch/1PL/2PL logLik differences above use engine-reported likelihoods, not a
new common-grid recalculation. Python library defaults use 21 GH nodes versus
R's 61 default nodes; native Rasch uses 61 GH. Constraints/optimization were not
fully aligned in these initial comparisons. This limits claims beyond the fixtures.
The 3PL result uses the later shared 61-GH target/grid/bounds profile, not the
less aligned initial 3PL result. Optimizers and stopping definitions still differ.

## 4PL interpretation

The independently started 4PL R solution stagnated near M12 u≈1. A single
subsequent R fit reused the **complete saved Python solution**; it is a dependent
optimization diagnostic, not another independent estimation experiment.
It retained bounds, priors, grid, algorithms and tolerances. R reached 3000 cycles
without convergence: **29/50** native changes were not below 1e-8; maximum change
**2.164495e-6**, final logLik increment approximately **2.3e-9**. R's common-grid
logLik exceeds saved Python by **.000228812638**. Close objectives do not erase
parameter/curve discrepancies or establish convergence/global optimality.

R's exact outer rule tests all absolute native parameter changes <1e-8 after
converting g/u to physical probabilities. It is not a gradient or logLik test.
The saved final pair confirms the rule failed; this supersedes earlier reports
where missing histories left final-pair satisfaction unresolved. The package
also forcibly reports false at its iteration cap. See [CONVERGENCE_TRACE.md](CONVERGENCE_TRACE.md).

The no-refit observed-information audit uses physical (a,b,c,u), excludes the
fixed mean/variance, and respects 11 active bounds. On the 37-coordinate active
face, the minimum eigenvalue is **.037891041708**, scaled condition number
**2349.497**, and the weakest physical eigenvector has **99.14% squared loading
on M12**. Two finite-difference steps support local numerical stability.
These are practical weak-identification/coupled-drift findings, **not structural
nonidentifiability**, exact rank deficiency, or global-optimum proofs. The
conditional feasible quadratic improvement estimate (~3.51e-6) is not a refit
or a bound on the unknown global maximum. Boundary regularity and SE/CI remain
uncertified. Full matrices, eigenvectors, numerical checks and limitations:
[observed_information/README.md](observed_information/README.md).

## Provenance and reproducibility

Actual engines: **R 4.6.1 / R mirt 1.48**, separately **Python 3.11.17 /
Python mirt 1.2.0**, NumPy2.4.6, SciPy1.17.1, pandas3.1.0. R is never substituted
with the Python package. Fixtures/settings/seeds, raw parameters, response traces,
versions, stdout/stderr and checksums remain in the original artifacts:

- Rasch/1PL/2PL: [run 40146456589](https://github.com/Tatung40/psikoanaliz/actions/runs/40146456589), commit0364494, artifact12581207421.
- Matched 3PL and original 4PL: [run 40149576359](https://github.com/Tatung40/psikoanaliz/actions/runs/40149576359), commit2b3f675, artifact12581672043.
- Warm-start R refit with traces: [run 40151828898](https://github.com/Tatung40/psikoanaliz/actions/runs/40151828898), commit498be38, artifact12582966948.
- Derived Hessian audit: commit3e4014c, committed matrices/report; no new fit.

Earlier failed artifacts, original estimates and `pending_report.json` are
preserved. The latter records historical preparation, not the current execution
status. Chronological audit documents retain their historical pending statements;
this README/final_summary.json provide the final classification. Existing
historical native R fixtures are not relabeled as new R executions.

The existing manual workflow currently schedules a single warm-start diagnostic;
it does not execute the whole table by default. Existing harness/reference scripts
and diagnostic reproduction commands remain available. **No further execution is
needed to reproduce this documentation from the saved evidence.** GRM/PCM/GPCM/
NRM/RSM independent R comparisons remain pending. Production estimators and
inference gates are unchanged; no fit result changes an inference safeguard.
These ML comparisons do not validate default MAP estimation, DIF error rates,
SE/CI coverage, or override existing sample-size restrictions.
