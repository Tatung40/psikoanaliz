# Formal DIF completion evidence

Baseline: `893d1cb` / PR #54 (2026-10-23); completed binary DIF work is retained. Preserve existing KTK, Rasch and binary MTK implementations.
This is a status record, not a claim that every family is validated for production DIF.

| Family | Existing calibrator / parameters and hypothesis | Identification / sample / anchors | Independent evidence / current status |
|---|---|---|---|
| Rasch | `rasch_dif`, concurrent MML; difficulty shift, Wald | Reference mean 0 and slope 1; reference SD and focal mean/SD separately free; >=50/group, warning <200; other-item anchors, purification | R fixtures and B9 tests; limited simulation, not a Type I/power study. Existing formal method retained. |
| 1PL | `irt_mg` / `irt_dif`; a=1, b shift, LR df=1 | Reference N(0,1); focal mean/SD free; >=50/group, warning <200; other-item anchors | Pinned R mirt 1.48 fixtures in B10, logL/LR/parameter tests. Predeclared fixed anchors now supported; excluded from testing. Existing R LR benchmarks retained; broad generalization remains unvalidated. |
| 2PL | Same; b shift LR df=1, a conditional on b LR df=1, omnibus df=2 | Same scale; a in [0.05,8], b in [-8,8] (`irt_mg` bounds); free focal impact; purified anchors | Fixed-anchor LR/scale/parameter tests match pinned R references. Unstable or insufficient purification blocks inference; selected-anchor p-values remain conditional. See the bounded fixed-anchor study below. |
| 3PL | `irt_mg34` / `irt_dif34`; a,b,c invariance | Reference N(0,1), focal impact free, invariant anchors; >=100/group, warning <300; all free parameters must be interior for LR inference | B11 boxed R mirt references. Boundary planted items are descriptive screens, not significant formal DIF. Broad simulation/coverage remains unvalidated. |
| 4PL | Same, plus upper asymptote d | Same linking; c<d and bounds; weak identifiability | B11 descriptive comparisons; UI inferential DIF disabled. Keep disabled pending valid boundary inference and independent simulation. |
| GRM | Python mirt pilot, ordered thresholds and slopes | Requires ordered category codes, common latent scale and invariant threshold/slope anchors | New isolated EXPERIMENTAL formal GRM: native concurrent MML; H2/H0 global, H2/Ha physical-threshold and H2/Hb slope LR; fixed anchors. Eight independent Python mirt 1.2.0 EM reference fits, analytic/integral/score/information/scale checks and 200 simulation repetitions. Qualified Turkish bundle available; R mirt, heterogeneous categories and general error control remain unvalidated. See GRM_DIF_VALIDATION.md. |
| PCM | Python mirt pilot, step intercepts with fixed slope | Ordered scores; step linking and free focal impact required | SCREEN ONLY. Need independent step parameterization / likelihood / df / recovery checks before formal integration. |
| GPCM | Python mirt pilot, steps and slope | Ordered scores; invariant anchors identify both location and scale | SCREEN ONLY. Need independent step/slope and nested multigroup validation. |
| RSM | Adapter lists RSM, but installed mirt 1.2.0 rejects this model name | Must impose shared category steps across items; cannot substitute unrestricted PCM | UNAVAILABLE for formal DIF. PR #23 confirms unsupported model. A tested shared-step constraint implementation or pinned R runtime is required. |
| NRM | Python mirt pilot; category-specific intercepts/slopes | Unordered labels; reference-category or sum-to-zero identification; invariant labels and anchors | Raw option-distribution screen only, explicitly NOT DIF. Need nominal multigroup identification, permutation-invariance and R reference tests. |

## Installed API audit

Python `mirt==1.2.0` is distinct from R mirt. Its installed public
`mirt.multigroup.fit_multigroup` and `multigroup_dif` APIs support several families;
the repository adapter currently calls single-group `fit_mirt` only. Source audit:
`multigroup_dif` exposes anchors, parameter families and reference group, and estimates
nonreference latent moments. Its internal `_likelihood_ratio_row` clips negative LR
to zero and records convergence separately from p-values. Do not expose its p-values
without independent regularity, convergence and nested-maximum validation.
The installed model-family list excludes RSM. `Rscript` is unavailable in this environment;
reuse existing pinned binary R fixtures rather than invent polytomous reference results.

## Current safety repair

`irt_dif` and `irt_dif34` now reject nonfinite log-likelihoods and an alternative
log-likelihood more than 1e-6 absolute units below the null. Only smaller roundoff
is clipped. Invalid comparisons return NaN, withhold p-values and explain the
optimization failure in Turkish. The tolerance is not scaled by N or logL.
The 3PL/4PL module's introductory inference policy now agrees with its existing
all-parameter boundary gate. New focused tests and existing R reference comparisons
are PR CI gates; this repair establishes no new model's scientific validity.

## Remaining acceptance work

Do not label all families FORMAL VALIDATED. General simulation coverage, independent
ordinal/nominal validation, probability-effect uncertainty and complete Turkish
DOCX/XLSX/CSV/300-dpi ZIP exports still require work. Binary fixed-anchor tests,
strict anchor-failure gates and pointwise parameter confidence intervals are implemented;
post-selection-adjusted inference remains unavailable. Existing exports/screens are retained;
no unvalidated formal polytomous p-values or automatic item deletion are introduced.

## Reviewed remote state

- Main `521fb31`: [full CI success, 530 passed / 2 skipped](https://github.com/Tatung40/psikoanaliz/actions/runs/39929140214).
- [PR #52](https://github.com/Tatung40/psikoanaliz/pull/52), merged: invalid nested maxima cannot produce p=1. Its tests are reused, not regenerated.
- [PR #20](https://github.com/Tatung40/psikoanaliz/pull/20): CI failed,
  `test_summary_tracks_failed_replicates`, `KeyError: elapsed_seconds`; mergeable=false.
  Unexecuted Monte Carlo infrastructure does not validate DIF.
- [PR #23](https://github.com/Tatung40/psikoanaliz/pull/23): CI failed,
  `test_polytomous_fit_model[RSM]`, unknown model RSM; mergeable=false.
  Smoke fits do not validate model parameters or DIF. Neither PR is merged.
- Issue #44 remains open for B14 scientific validation and reports.

Next scientific priority (user order): PCM, then GPCM/RSM/NRM, then the remaining
3PL/4PL inference gaps. GRM remains explicitly experimental outside the tested design;
its physical-threshold hypothesis must not be confused with slope-intercept equality in R mirt. Keep the installed unvalidated API separate from formal
production inference; do not merge the failed PRs as a substitute for validation. See `DIF_CHECKPOINT.json` for execution evidence.

## Binary fixed-anchor contract (current branch)

- `fixed_anchors=[...]`, `purify=False`: declared invariants are never tested or freed;
  other non-anchor items remain group-specific nuisance parameters in nested fits.
  Independent content/prior-study evidence is required; selecting them from these data
  does not turn the analysis into predeclared inference.
- Every studied item must retain at least `max(2, ceil(.2*k))` valid anchors (configurable
  integer >=2). This is a conservative operating rule, not a universal sample/anchor
  adequacy theorem. Constant or unobserved items cannot supply anchors.
- Purification must stabilize with valid tests. Cycles, an exhausted round budget,
  insufficient per-item anchors, and changed final decisions stop the analysis.
- Final LR inference requires converged, interior nested fits with valid observed
  information. Missing responses are skipped under MAR, on the same respondent subset
  and missingness pattern for all nested fits; exclusions and missingness are explicit.
- Omnibus, uniform and nonuniform tests have separate BH families over tested items;
  no joint FDR guarantee is asserted across those families. Type labels use the
  nonuniform BH result and the conditional IRF crossing. Parameter intervals are
  pointwise 95% Wald intervals; probability-scale intervals remain pending.
- Pinned R mirt 1.48 fixtures: single-item uniform LR agrees within 1e-4 (1PL) /
  1e-5 (2PL); fixed-anchor joint slopes/difficulties and focal moments agree within
  the existing 2e-3 parameter tolerance (variance 3e-3). The fixture anchor list used
  for the benchmark was historically selected; numerical agreement alone is not
  evidence of predeclaration for those historical data.

## Bounded fixed-anchor simulation

Runner: `scripts/dif_fixed_anchor_simulation.py`; cache uses source/version/config
SHA-256 and deterministic independent repetition seeds. No raw responses are exported.
Design: 12 binary 2PL items, reference N=800, theta~N(0,1); focal N=500,
theta~N(-.4,1.2^2); 10 genuinely invariant, predeclared anchors; two studied items.
Null: all parameters invariant. DIF: M3 delta-b=.6, M7 delta-a=-.9.
Each condition has 100 repetitions. Exact Clopper-Pearson intervals quantify Monte
Carlo uncertainty; rates and pointwise interval coverage condition on valid fits,
with inconclusive counts reported separately. Under the complete null, the probability
of any BH rejection estimates FDR for these two studied items. It is not a general
FDR, Type I, power, or coverage certification across other designs or selected anchors.
Results are recorded in `tests/fixtures/irt_b10/fixed_anchor_simulation_summary.json`.

Observed study results (100 valid / 100 total in each condition, zero inconclusive):

| Quantity | Estimate | Exact 95% Monte Carlo interval |
|---|---:|---:|
| Complete-null: any BH rejection | 7/100 = .07 | [.0286, .1389] |
| Complete-null: M3 BH rejection | 3/100 = .03 | [.0062, .0852] |
| Complete-null: M7 BH rejection | 4/100 = .04 | [.0110, .0993] |
| Planted uniform M3: BH rejection | 97/100 = .97 | [.9148, .9938] |
| Planted nonuniform M7: BH rejection | 100/100 = 1.00 | [.9638, 1.00] |

Pointwise contrast-interval coverage was 94–98% across the two parameters/items and
conditions; each estimate has its own exact interval in the JSON artifact. Null focal
mean/SD RMSE: .0861/.0693; planted DIF: .0718/.0718. All 200 repetitions used 1,000
optimizer fits (~114 seconds on this environment), with 4 initial repetitions reused
from cache. The null interval is wide and permits material inflation; **nominal FDR
control is not established**. More precise, broader design evaluation remains pending.

CSV/XLSX item exports escape formula-like labels, include the actual per-test anchors,
likelihoods, raw/adjusted p-values, parameter intervals and decisions, and exclude
respondent rows. Full academic DOCX/ZIP packages and probability-effect uncertainty
remain unfinished.

## GRM evidence added after PR #54

See `GRM_DIF_VALIDATION.md` for exact hypotheses, parameterization, numerical tolerances,
Monte Carlo intervals and remaining limitations. The new page requires an explicit
experimental/ordinal-data acknowledgment; it does not change binary DIF or rename the
existing score screen. Its ZIP includes qualified DOCX, XLSX, BOM CSV and 300-dpi curves,
without respondent records. Completed binary numerical gates are selected only when their
sources/shared helpers change; the final main regression still runs every test.
