# Polytomous IRT source audit and independent R strategy

Reference: PR #63, `18d1bcf72c85a174ad278ae4d8ea925b412c0275`.
Report-only audit: no new models, calibration fits, simulations or production changes.
Independent **R** comparisons for these five models remain pending. Existing Python
recovery, algebra and browser checks are evidence of their stated scope, not R validation.

## 1. Actual implementation matrix

Scope: the single-group application calibration paths; additional DIF engines are
identified separately below. “Implemented” does not mean scientifically certified.

| Model | Verified execution path/status | Estimation, identification and actual bounds | R mirt classification |
|---|---|---|---|
| GRM | Implemented via `core.irt_b14.fit_b14` → `core.irt_python_mirt_adapter` → Python mirt1.2.0 `GradedResponseModel` | Marginal ML/EM, standard normal mean0/variance1; a∈[.1,5], physical thresholds b∈[-6,6], adjacent threshold gap≥1e-6 | **TRANSFORMATION REQUIRED**: R `graded`, b_s=-d_s/a; match ordering, gaps and coupled boxes |
| PCM | Implemented via same adapter, `PartialCreditModel`, fixed discrimination mask/canonical a=1 | Marginal ML/EM, mean0/variance1, item-specific adjacent steps∈[-6,6]; steps need not be ordered | **TRANSFORMATION REQUIRED**: R `gpcm`, a1 fixed1, variance fixed1; differences of cumulative category intercepts |
| GPCM | Implemented via adapter, `GeneralizedPartialCredit` | Marginal ML/EM, mean0/variance1, a∈[.1,5], adjacent steps∈[-6,6]; no step-order constraint | **TRANSFORMATION REQUIRED**: R `gpcm`; cumulative intercept/adjacent-step mapping and coupled bounds |
| RSM | Implemented in **native** `core.irt_rsm.fit`; public B14/Python `fit_mirt` route is **NOT IMPLEMENTED** for RSM and is rejected | Marginal ML, analytic score/L-BFGS-B; slope1, mean0, default variance estimated; sumτ=0; δ∈[-8,8], free τ∈[-6,6], last τ=-Σfree (not separately boxed), σ∈[.05,10]; optional unit variance | **TRANSFORMATION REQUIRED** for native RSM versus R `rsm`, including common-step origin/variance and feasible-region checks; B14 library route **NOT IMPLEMENTED** |
| NRM | Implemented via adapter, `NominalResponseModel`; explicit `se_method='oakes'` | Marginal ML/EM, mean0/variance1; category0 slope/intercept fixed0; free effective slopes∈[-5,5], intercepts∈[-6,6] | **TRANSFORMATION REQUIRED** on a regular R nominal chart; **NOT COMPARABLE** if required anchor contrast is singular or equivalent constraints cannot be represented |

Python library calibration defaults are EM21-point Gauss–Hermite, max500,
absolute marginal logLik increment <1e-4; no item priors in this ML route.
Calling it “MAP” because 3PL/4PL have another application option would be wrong.
Library item optimizer/stability details differ from the outer convergence flag.
Native RSM defaults: GH41, max5000, L-BFGS-B ftol1e-15/gtol1e-12. Its *reported*
convergence requires finite objective, score/person≤1e-6 and no box-bound parameter
(within 1e-6); optimizer success alone is insufficient.

PCM also exists inside native RSM as its nested LR alternative: item steps∈[-14,14],
with the same free-variance/default latent specification. That is not the B14
unit-variance PCM target and must not silently substitute for it.

### Probability and information equations (unidimensional, D=1)

Let categories be c=0,…,C−1, L(z)=logistic(z), P0 baseline implicit where appropriate.

- GRM: S_s=L[a(θ−b_s)], S_0=1, S_C=0; P_c=S_c−S_{c+1}.
- PCM/GPCM: P_c∝exp[Σ_{s=1}^c a(θ−b_s)], with a=1 in PCM and empty sum0.
- RSM: same formula with a=1 and b_is=δ_i+τ_s, shared τ across items, Στ=0.
- NRM: P_c=exp(A_cθ+D_c)/Σ_h exp(A_hθ+D_h), A_0=D_0=0. No correct-option
  key or success ordering is inferred from category labels.

For every model I_i=Σ_c(P'_c)²/P_c; test information is Σ_i I_i. This reduces to
Var(category) for PCM/RSM, a²Var(category) for GPCM, and Var(A_category) for NRM.
The library's `information(theta)` without item_idx already returns test information:
do not sum that result again. `core.irt_b14_persons_info.independent_information`
provides the separate probability-derivative check.

### SEs, tests and limits

GRM/PCM/GPCM library auto-SE uses its observed-information/Oakes route (documented
implementation uses Louis identity). NRM auto instead selects complete-data
curvature; the adapter explicitly requests Oakes because auto previously severely
undercovered. This is an algorithm/path discrepancy, not a parameter-label issue.
Native RSM inverts a finite-difference observed-information matrix only after
convergence and its positive-definiteness guard; otherwise SEs are withheld.
EAP posterior SD, conditional SEM and item parameter SE are different quantities.

Existing evidence: `validation/POLYTOMOUS_MC_VALIDATION.md` and
`validation/RSM_VALIDATION.md`; tests `test_irt_b14_persons_info.py`,
`test_irt_b14_model_curves.py`, `test_nrm_se_fix.py`, `test_irt_rsm.py`,
`test_python_mirt_rsm_support_gate.py`, plus browser CSV/XLSX comparisons.
GRM/PCM/GPCM recovery/coverage evidence is bounded to its registered designs;
NRM's explicit-SE adapter evidence does not validate auto-SE. Native RSM's
independent-library comparison was **Python**, not R; N300/k20 location coverage
was below its registered interval. Item-fit outputs and broader population,
missingness, dimensionality or boundary behavior are not generally certified.

Additional native fixed-anchor two-group engines exist in `core.grm_dif` and
`core.polytomous_dif` (PCM/GPCM/RSM/NRM): separate L-BFGS-B concurrent MML,
anchor/latent constraints and quadrature audits, with different parameter bounds.
The existing single-group R harness is **NOT COMPARABLE** to their DIF hypotheses
without a separate multiple-group design. This audit does not certify DIF size,
power, effect intervals or purification by transferring single-group evidence.

## 2. R compatibility: representation versus genuine discrepancies

R 1.48 source checked: `cran/mirt`, tag1.48, `R/06-LoadPars.R` and documented mirt
families. R `graded` uses slope/intercepts. R `gpcm` fixes category scores ak=0…C−1,
baseline d0=0, and estimates cumulative category intercepts D_s. Thus
b_s=−(D_s−D_{s−1})/a, **not** −D_s/a. The current reference script's probability
contrast method avoids assuming the raw coefficient layout.

Matching physical feasible regions is essential:
- GRM: −6a≤d_s≤6a, d_s−d_{s+1}≥a×1e-6, .1≤a≤5.
- PCM: a=1, −6≤D_s−D_{s−1}≤6; normal mean0/variance1 fixed. Using ordinal
  R `Rasch` with a free latent variance would compare a different model.
- GPCM: −6a≤D_s−D_{s−1}≤6a plus slope bounds. Independent cumulative-intercept
  boxes are not equivalent. These are expressible linear inequalities in R native
  coordinates, but require actual R layout/Jacobian preflight before fitting.
- Native RSM: compare the physical b_is matrix first; obtain δ_i by centering
  shared steps, verify equal τ across items and match free σ. Transform the native
  free-τ box and derived-last-τ constraints too. A unit-variance RSM fit is not the
  default free-variance native model. Exact R constraint equivalence is **UNKNOWN**
  until its initialized layout and probability contrasts are checked at runtime.
- NRM: R nominal uses effective A_c=a1×ak_c and baseline intercept contrasts.
  It fixes ak0=0 and ak_last=C−1. If A_last=0 while another contrast is nonzero,
  that finite chart cannot represent the same coefficients. Even on a regular chart,
  match effective-slope bounds, baseline and predeclared θ reflection; never sort
  nominal options by “difficulty”. Use the full covariance Jacobian for any SEs.

All these models have comparable category probabilities/information **after**
scale, scoring and constraints match. Raw parameter/SE mismatch alone is not
proof of an algorithm defect. R's default equally spaced quadrature is not
Python GH simply because node counts agree. R's strict parameter-change rule
(converting g/u where applicable) differs from Python's logLik-increment rule;
same numeric TOL is not an identical stopping criterion. Compare respondent
marginal likelihoods on a single explicit normal GH grid; retain backend-reported
logLik/AIC/BIC separately to expose constants/grid differences.

## 3. Smallest GRM → PCM independent validation plan

Reuse PR63 `validation/r_mirt_phase1/{harness.py,reference.R}` and fixed
`validation/evidence/p5_browser/poly_ordinal.csv` (500×8, categories0…3;
original browser generator seed1). Reference seed20261024; record data/script
SHA256 and exact R/Python versions. No new simulations. This GRM-generated
fixture is a cross-engine optimization comparison for PCM too, **not** PCM
parameter-recovery evidence. Reject missing/noninteger/empty-category data;
never silently reorder, merge or impute. The B14 adapter requires every declared
category in each item, in addition to global category checks.

1. **No-fit preflight first:** R mirt1.48 `pars='values'` for graded/gpcm; verify
   parameter names, fixed latent layout, score anchors and free-vector Jacobians.
   Probe category equations, information, boundaries and GRM gap constraints at
   conventional deterministic parameters. No fitted Python coefficients supplied.
2. Add an opt-in *polytomous* profile to the existing harness/reference (future
   work only). Existing `--matched` is restricted to 3PL/4PL and **must not** be
   reused unchanged. Preserve every binary result and current single-refit workflow.
3. GRM first, PCM second: one independent fit per engine/model only after preflight.
   Mean0/variance1/D1/ML/no priors, same explicit 61-GH nodes/normal weights, conventional
   predeclared starts, recorded caps/tolerances and per-engine stopping definitions.
   Use the supported R inequality API with the above full feasible region. If any
   constraint cannot be matched, label NOT COMPARABLE and stop before fits.
4. Compare transformed item parameters, category probabilities on the saved
   [-4,4] grid, per-item/test information, and stable log-sum-exp respondent
   common-grid logLik. Keep convergence traces, boundary flags, physical feasible
   scores, raw coefficients, versions/settings, stdout/stderr and all failures.
   Do not equate two convergence flags with equal maxima.
5. Point estimates first. SE/CI remain NOT COMPARABLE until matching estimator
   definitions, identification, full covariance Jacobians and boundary regularity
   are verified. No scalar rescaling of marginal SEs; no new coverage claim.

Result rules: **PASS** requires matched specification, demonstrated convergence
and pre-registered scientifically justified cross-engine discrepancy criteria;
none are established yet, so a scientific PASS is currently unavailable.
**PARTIAL** means valid executed comparisons with finite recorded discrepancies
but uncalibrated acceptance or remaining optimizer/SE limitations. **FAIL** retains
execution/schema/invalid-probability/constraint/nonconvergence failures with raw
outputs. **INCONCLUSIVE** means an unresolved identification/mapping, derivative
instability or insufficient evidence; do not reinterpret missing execution as PASS.
Existing Monte Carlo recovery cutoffs are not cross-engine numerical tolerances.

## 4. Verification and exact next executable step

Verified repository routes and installed **Python mirt1.2.0** model/estimation
source, not UI labels. Model source `models/polytomous.py` SHA256:
`9a0c186c2cab1d6ff9b1f8597281a02f72c8d8a6b80bec54967033ce8ca7bdce`.
Library code also inspected: `_factory.py`, `estimation/{base,em,quadrature,_graded_order}.py`.
No dependency installation. Only no-fit existing tests executed:
`test_irt_b14_model_curves.py` and `test_python_mirt_rsm_support_gate.py`: **3 passed**.

Next executable action, **only in an existing R environment**, creates layout
specifications without fitting. Do not run the full current workflow for this:

```sh
Rscript -e 'library(mirt); stopifnot(as.character(packageVersion("mirt"))=="1.48"); x<-read.csv("validation/evidence/p5_browser/poly_ordinal.csv"); for(t in c("graded","gpcm")) write.csv(mirt(x,1,itemtype=t,pars="values",verbose=FALSE),paste0("/tmp/",t,"_values.csv"),row.names=FALSE)'
```

Then review those layouts and implement only the validation-profile preflight
specified above in a separately authorized change. This audit did not execute R,
new fits, simulations or a full regression, and did not modify PR63/evidence.
