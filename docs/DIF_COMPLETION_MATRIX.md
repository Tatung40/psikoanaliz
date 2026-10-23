# Formal DIF completion evidence

Baseline: `4558b0d` (2026-10-23). Preserve existing KTK, Rasch and binary MTK implementations.
This is a status record, not a claim that every family is validated for production DIF.

| Family | Existing calibrator / parameters and hypothesis | Identification / sample / anchors | Independent evidence / current status |
|---|---|---|---|
| Rasch | `rasch_dif`, concurrent MML; difficulty shift, Wald | Reference mean 0, common free SD; focal mean/SD free; >=50/group, warning <200; other-item anchors, purification | R fixtures and B9 tests; limited simulation, not a Type I/power study. Existing formal method retained. |
| 1PL | `irt_mg` / `irt_dif`; a=1, b shift, LR df=1 | Reference N(0,1); focal mean/SD free; >=50/group, warning <200; other-item anchors | Pinned R mirt 1.48 fixtures in B10, logL/LR/parameter tests. Limited simulation; fixed-anchor calibration and broad Monte Carlo still needed. |
| 2PL | Same; b shift LR df=1, a conditional on b LR df=1, omnibus df=2 | Same scale; a in [0.15,4], b in [-4,4]; free focal impact; purified anchors | Existing R comparisons and planted/null tests retained. Post-selection p-values and low simulation replication limit inference. |
| 3PL | `irt_mg34` / `irt_dif34`; a,b,c invariance | Reference N(0,1), focal impact free, invariant anchors; >=100/group, warning <300; all free parameters must be interior for LR inference | B11 boxed R mirt references. Boundary planted items are descriptive screens, not significant formal DIF. Broad simulation/coverage remains unvalidated. |
| 4PL | Same, plus upper asymptote d | Same linking; c<d and bounds; weak identifiability | B11 descriptive comparisons; UI inferential DIF disabled. Keep disabled pending valid boundary inference and independent simulation. |
| GRM | Python mirt pilot, ordered thresholds and slopes | Requires ordered category codes, common latent scale and invariant threshold/slope anchors | SCREEN ONLY in current DIF UI. Installed multigroup API is a candidate, not independently validated here. Need nested threshold/slope fits, constraints, recovery and pinned R comparisons. |
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

Do not label all families FORMAL VALIDATED. Fixed-anchor inference versus selected
anchors, failure with insufficient/unstable anchors, sufficiently replicated null/DIF
simulations with Monte Carlo intervals, effects with uncertainty and complete Turkish
DOCX/XLSX/CSV/300-dpi ZIP exports still require work. Existing exports/screens are retained;
no unvalidated formal polytomous p-values or automatic item deletion are introduced.

## Reviewed remote state

- Main `4558b0d`: [CI success](https://github.com/Tatung40/psikoanaliz/actions/runs/39906198907).
- [PR #20](https://github.com/Tatung40/psikoanaliz/pull/20): CI failed,
  `test_summary_tracks_failed_replicates`, `KeyError: elapsed_seconds`; mergeable=false.
  Unexecuted Monte Carlo infrastructure does not validate DIF.
- [PR #23](https://github.com/Tatung40/psikoanaliz/pull/23): CI failed,
  `test_polytomous_fit_model[RSM]`, unknown model RSM; mergeable=false.
  Smoke fits do not validate model parameters or DIF. Neither PR is merged.
- Issue #44 remains open for B14 scientific validation and reports.

Next: reproduce the documented blockers on isolated branches, validate the installed
multigroup API against pinned independent ordinal/nominal references, then integrate only
the independently verified subset. See `DIF_CHECKPOINT.json` for execution evidence.
