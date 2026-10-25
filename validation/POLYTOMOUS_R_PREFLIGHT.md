# GRM/PCM no-fit preflight — PR65

Current **R execution: NOT EXECUTED** (Rscript unavailable in this workspace).
Machine-readable actual preparation result: `evidence/poly_r_preflight/report.json`.
No model fits, simulations, package installations or scientific PASS. PR63 branch,
its harness and evidence are unchanged. This implements only the preflight from
`POLYTOMOUS_R_MIRT_AUDIT.md`, not the future independent-fit experiment.

## Reproduction

Use an existing Python environment containing the repository dependencies and
Python mirt1.2.0. The PR63 Git object must already be present:

```sh
git cat-file -e 18d1bcf72c85a174ad278ae4d8ea925b412c0275:validation/r_mirt_phase1/harness.py
PYTHONPATH=. python validation/poly_r_preflight.py --output /tmp/grm-pcm-preflight-new
```

The driver does not fetch code or install dependencies. In a fresh/shallow clone,
first obtain the pinned PR63 commit through your normal repository workflow.
It loads the exact local Git helper source (hash recorded), reusing PR63
`semantic_parameters`, `compare_curves` and its declared fixture path rather
than copying their implementation or importing a binary-fit execution path.
Do not run the existing PR63 workflow: it schedules an R refit, not this preflight.

When an existing R+mirt1.48 runtime is available the same command independently
executes `Rscript validation/poly_r_preflight.R OUTPUT`. To inspect prepared
inputs on another R host, copy the prepared directory and run that R command;
then run the Python driver in a new output directory on the joint environment
to produce the complete cross-engine comparison. Do not relabel an isolated R
invocation or a missing R process as a completed comparison.

## What executes, without fitting

- Fixed existing `poly_ordinal.csv` (500×8, codes0…3); every item must contain all
  categories. No response generation, recoding, imputation or estimated parameters.
- Conventional physical probe parameters; PCM deliberately includes unordered
  adjacent steps. Python constructs existing GRM/PCM item classes directly.
- R `mirt(..., pars='values')` initializes layouts only. It verifies actual names,
  normal mean0/variance1, GRM ordering, gpcm score anchors0…3/d0=0 and PCM a1=1
  fixed. Initial and configured values tables are preserved separately.
- Forward/inverse physical/native parameter Jacobians are checked analytically
  and by central differences. GRM uses d_s=-a*b_s; PCM uses D_s=-Σ_{h≤s}b_h.
- Explicit linear constraints match physical boxes and the GRM gap1e-6; their
  free-coordinate Jacobians and feasible/infeasible witnesses are checked. PCM
  steps are never sorted. A mismatch stops R immediately with checked diagnostics.
- Native R S4 graded/gpcm objects feed package `probtrace` and `iteminfo` without
  calibration. The constructor pattern was checked against **cran/mirt tag1.48,
  R/simdata.R**, but that simulator is NEVER called. Raw item vectors come from
  the verified values layout. Independent closed forms cross-check both outputs.
- Both languages receive identical 61-GH nodes/normal weights. R checks moments
  and its actual `makeopts` handling of customTheta/customPriorFun. No EM is
  executed: likelihood integration by a fitted estimator remains unverified.
- PR63 helpers compare keyed probabilities and item/test information. Comparison
  precision is base R all.equal's default sqrt(machine epsilon), used only for
  fixed-parameter algebra; no estimation/scientific acceptance tolerance changed.

Each run records versions, scripts/helper/data SHA256, deterministic seed20261024,
physical probes, GH/display grids, Python/R curves, R parameter layouts, Jacobians,
constraint matrices, check results, sessionInfo and stdout/stderr. No covariance
or SE comparison is implied. R nonzero exits/timeouts are FAIL and preserve logs;
missing R is NOT EXECUTED. Full checks can only yield **PREFLIGHT CHECKED / NOT
SCIENTIFICALLY VALIDATED**, never a scientific PASS before independent fits.

## Actual checks and unresolved risks

Python fixed-probe mapping errors: GRM and PCM **4.4408920985e-16**. Probability
sum errors: GRM2.22e-16, PCM4.44e-16. Existing curve/gate tests plus new no-fit
and error-preservation tests: **5 passed**; final targeted recheck of the two new
tests also passed. The mocked R-error test is not an actual R execution.

All R runtime layout/constructor/Jacobian/constraint/information and cross-engine
results remain **NOT EXECUTED**, not verified by source inspection alone. Numerical
behavior at exact box witnesses is not silently relaxed. GH API configuration
checks cannot establish actual EM behavior or equal stopping criteria. SE/CI,
convergence at fitted estimates and scientific discrepancy criteria remain open.

**Next action:** execute the driver once in the provisioned R1.48/Python environment,
inspect its R check files and report, and stop on any mismatch. Only a separate
subsequent authorization should schedule independent GRM/PCM fits.
