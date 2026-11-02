# RESOURCE-LIMITED TECHNICAL VALIDATION — 30 paired seeds

**NUMERICALLY VALIDATED — 30 PAIRED SEEDS**  
**FWER NOT CERTIFIED**  
**SCIENTIFICALLY NOT CERTIFIED**

Original100-pair Stage A: **USER-STOPPED / INCOMPLETE**. User requested the resource/Codex-quota stop; no statistical-success stopping. Protocol f4f3c237 and frozen runner385e4cc were not changed. User amendment a38a92f is separate. PR127 stays stacked on PR126/cfbb199; do not merge.

## Preservation and boundary

First disk inspection found49 completed/valid cases, with50 in progress (the old37 update was not assumed current). By installation of the boundary controller,53 were complete. The unchanged original process completed only seven further cases. A progress-output FIFO barrier held the process after fsynced record60 and before the next case; termination happened only at that boundary. Log output ends at59 because record60 precedes the blocked progress/log write. No61st optimizer start or excess completed case exists.

Raw journal and control snapshots remain on disk unchanged. Canonical gzip archives round-trip to the exact journal/prefix SHA256; uncompressed copies are deliberately not duplicated in Git. Original metadata/progress/active-model/log snapshots and their original SHA256 manifest are retained. No completed case was rerun, deleted, replaced or silently relabelled.

Seeds2026110100–2026110129, original minus-then-zero order: **30/30 complete and valid pairs;60/60 valid datasets;0 invalid,0 missing in this resource sample,0 excess**. Original200 plan:60 attempted/valid,140 not run. The original preflight had reconstructed200 same-hash matrices before the user stop; the other140 matrices are unused input artifacts, not completed analyses.

Pilot fits were not pooled because runtime differs (StageA Python3.12/NumPy2.5.3/SciPy1.18.1/pandas3.0.5 versus archived pilot3.11/2.4.6/1.17.1/3.1.0). Response matrices and cached ordinal decisions match original per-seed SHA256. No GH61 or ordinal calibration was rerun.

## Numerical gates and resources

| Check | Observed |
|---|---:|
|Native-converged datasets / successful unique optimizations|60/60; 420/420|
|Verified real cold starts|420|
|Maximum normalized GH101 score|7.32227218e-07 (<2e-6)|
|Maximum normalized GH161 score|7.32201749e-07 (<=1e-5 unchanged)|
|Minimum native / GH161 information eigenvalue|6.53999205 / 6.53999226|
|Maximum GH161 information condition|168.548330|
|Maximum GH101/GH161 logLik discrepancy|1.72143189e-07 (<=1e-3)|
|Maximum LR grid difference|1.26397936e-07|
|Case wall sum / approximate original-stage boundary wall|1305.280s / 1309.066s|
|Native fitting / independent audit / postprocessing|338.428s / 270.210s / 277.168s|
|Approximate total wall including final audit|1586.234s (26.44min)|
|Analysis / postprocess process RSS high-water|152.793 / 342.406MiB|

Wall to boundary is last durable59 progress elapsed plus recorded case60 wall, excluding subsecond final ledger/control overhead; it is labelled approximate. RSS values are cumulative per-process high-water, not summed simultaneous memory. Both phases remain under original7200s/1024MiB limits.

All420 hypotheses preserve P01–P08 anchors, referenceN(0,1), free focal mu/SD, encoded bounds, positive ordered thresholds, native covariance/relative-PD rules and omnibus/threshold/slope df4/3/1. H2 is shared once per dataset (480 reported slots are not480 independent replications). Every fit is native start_fit=None, GH101 only; no warm starts, rescue or density escalation.

## Descriptive FWER/FDR and uncertainty

Primary omnibus P09/P10 family has Holm alpha=.05 and BH q=.05. Under this global latent null BH event probability equals FDR. Threshold and slope families are separate diagnostics. These are exploratory descriptions after the resource amendment, not a completed100-pair preregistration or nominal-control certificate.

|Impact|Holm events / n|Holm-FWER|BH event/FDR|MCSE|Marginal95% Clopper–Pearson|
|---|---:|---:|---:|---:|---|
|minus|2/30|0.066667|0.066667|0.046321|[0.008178, 0.220735]|
|zero|1/30|0.033333|0.033333|0.033333|[0.000844, 0.172169]|

CP97.5 and inherited joint11 component intervals are in resource30_summary.json. Joint11 scope is ONLY two GRM Holm rates plus nine contrasts; extra BH/ordinal rate interval fields are not an expanded jointly covered family. MCSE uses sample SD/sqrt(n); a zero plug-in MCSE does not mean zero uncertainty. Original100-denominator unknown outcomes remain in summary.json; resource-sample bounds use30, never missing=no-event.

Cached same30-seed ordinal Holm events: minus total1/30, rest5/30; zero total1/30, rest1/30. No score-method result outside these30 seeds enters resource comparisons. Observed-score and latent-invariance nulls differ; no universal method-superiority claim.

|Paired Holm-event contrast|n|Difference|MCSE|Conservative exact-component95% interval|
|---|---:|---:|---:|---|
|GRM-minus-total_minus|30|0.033333|0.058395|[-0.251718, 0.287583]|
|GRM-minus-rest_minus|30|-0.100000|0.073500|[-0.371081, 0.197225]|
|GRM-minus-total_zero|30|0.000000|0.000000|[-0.135901, 0.135901]|
|GRM-minus-rest_zero|30|0.000000|0.000000|[-0.135901, 0.135901]|
|GRM_zero-minus-minus|30|-0.033333|0.033333|[-0.194587, 0.189722]|
|total-minus-rest_minus|30|-0.133333|0.063124|[-0.332565, 0.110165]|
|total-minus-rest_zero|30|0.000000|0.000000|[-0.135901, 0.135901]|
|(GRM-total)_zero-minus-minus|30|-0.033333|0.058395|[-0.467483, 0.439752]|
|(GRM-rest)_zero-minus-minus|30|0.100000|0.073500|[-0.396665, 0.551373]|

All paired intervals include0. Seven binary McNemar diagnostics: Holm7 and BH7 adjusted p>=.875; McNemar does not apply to the two interactions. Joint11 Hoeffding contrast intervals, discordant/16-cell tables and failure bounds are retained in JSON. Neither these observations nor numerical convergence certifies FWER.

## Independent audit, evidence and CI

Raw reader independently checked 420 starts, 840 direct-row likelihoods, 420 GH161 gradient vectors, 420 stored/reconstructed information matrices and 360 LR records on both grids. Full-coordinate two-step direct-row FD results were reused and checked; no optimizer was called by the audit. Source/input/raw/gzip SHA256 and all100 preflight pairing traces were checked; the scientific sample remains30 pairs. Resource30 statistical values and denominator transformation passed an independent count/CI audit.

Evidence: results/records.jsonl.gz, results/metadata.json, results/summary.json (original incomplete plan), results/resource30_summary.json, results/audit.json, results/resource30_audit.json, results/SHA256.json, same-seed matrices, ordinal.json and user_stop_snapshot/. Frozen-source hashes are in metadata; new postprocessing hashes are separate. Core/production/p-suppression files remain unchanged.

Targeted local validation:10 passed,0 skipped,0 failed, no real optimizer. New targeted GitHub evidence CI will be reported on the final commit. Most recent completed general CI on385e4cc:2 failed,660 passed,14 skipped. Failing unchanged archived tests: test_gh_reoptimization::test_cached_reoptimization_and_dense_gates_without_calibration and test_grm_gh101_uniform_pilot::test_saved_uniform_pilot_and_independent_audit_no_calibration. Recorded mismatch includes1.39e-17 in exact cached cold-start reconstruction and derived hash identities. Basecfbb199 CI had passed; current cross-platform exact-bit CI reproducibility is unresolved, not declared a pre-existing green guarantee. No tolerance was loosened or failed test skipped.

No100/500/2500 continuation, automatic merge or scientific certification is authorized.
