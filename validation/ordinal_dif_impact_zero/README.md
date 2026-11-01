# Impact0 ordinal DIF control — opt-in, paired with immutable PR120

Plan committed before new outcomes: `88ba781`; execution code committed before fitting: `7f32a58`. Base main `538e974` includes PR120. Production estimators, inference gates and labels are untouched.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 python -m validation.ordinal_dif_impact_zero.experiment
```

The existing PR120 runner/production ordinal LR are reused through an explicit validation-only adapter changing **only impact to0**. Inputs remain N500/500, J10, C4, invariant latent GRM, P09/P10 family, total/rest conditioning, Holm .05 and secondary BH. No independent R, GRM-LR, 4PL calibration or old negative-impact fit is executed. The runner refuses to overwrite evidence. Routine CI only runs lightweight tests.

## Pairing and storage

Seed equality alone is not accepted. Before fitting each control dataset, trace the actual generator's two normal and ten item-uniform draws. Require the archived generator/production source hashes and package versions. Reconstruct negative-impact responses using these identical draws with focal theta shifted by−.5, and require the exact archived response+group content hash. Also verify identical reference rows and monotone focal responses. This is a deterministic replay of the DGP, **not a repeated negative-impact calibration**. Old estimates/p values/decisions are read from the immutable archive. Source/version/seed replay establishes draw identity; response equality alone would not establish it.

Preserve every control repeat's raw LR/p/decisions, seed, data hash and timing; first30 and failed matrices are NPZ. All-repeat `crn.jsonl.gz` records draw hashes/layout, reference equality and archived response-hash checks. Metadata records actual impact, sources, versions, plan and archive checksums. `pr120_manifest.json` protects39 existing PR120 files before/after execution. No negative evidence is rewritten.

## Preregistered uncertainty and multiplicity

Primary Δ0=T0−R0 for paired Holm events. Secondary changes T0−Tm and R0−Rm (m means impact−.5). Report b/c discordances, exact McNemar, paired sample-SD/√n MCSE, conservative exact paired95% interval and marginal95% Hoeffding interval for each binary contrast. Exact paired interval uses π=P(discordance), q=P(positive-only|discordance), Δ=π(2q−1) and the two97.5% CP rectangle. Zero discordance is not certainty.

Interaction W=(T0−R0)−(Tm−Rm) uses the **same dataset-level quadruples**, not independent variances. W∈[−2,2]. Preserve all16 binary cells and3×3 signed-event cells. Report sample-SD/√n MCSE and Hoeffding interval with range4. Its supplementary conservative exact95% interval subtracts two97.5% exact paired component intervals (each constructed from98.75% CP intervals), valid by union bound without assuming independence. **McNemar does not apply to W.**

Individual95% contrast intervals are marginal, **not joint95%**. Additionally report **Bonferroni simultaneous95% Hoeffding intervals over all FOUR contrasts**, alpha=.05/4 each; width2 for binary differences, width4 for interaction. Report Holm-adjusted McNemar p across the THREE binary contrasts separately. McNemar tests equal event probabilities, never nominal FWER control.

Each method's FWER has97.5% CP bounds, giving joint≥95% within a condition; these are not simultaneous across both conditions/all contrasts. BH intervals are a separate descriptive family. BH event probability is FDR only in this global latent-null design; raw itemwise rates use dataset-clustered MCSE. No mixed-null/power/selection claim.

## Stopping and interpretation

First30 assess validity/runtime only: continue iff30/30 valid and projected2500 cost≤1800 seconds. The same30 count toward fixed2500; no replacements, retries or result-based changes. All failures remain in raw records; incomplete families invalidate their method, missing contrasts receive conservative unknown-outcome bounds. Confirmatory results require all2500 relevant complete pairs; otherwise INCONCLUSIVE. A source/RNG/hash mismatch is BLOCKED.

Primary direction follows the original marginal95% Hoeffding rule after full valid fixed-R completion; joint cross-contrast conclusions use the simultaneous intervals. Scoped inflation requires lower FWER CP>.05, scoped control upper≤.05, otherwise inconclusive. A point estimate near .05 is not equivalence or control certification. No arbitrary equivalence margin is introduced.

Any causal interpretation is restricted to the controlled focal latent-mean change in this DGP. CRN improves simulation pairing; it is not a randomized empirical human-treatment study. Total/rest match on different observed variables and have different conditional nulls. Latent invariance does not guarantee either conditional null. Results do not certify other DGPs, item families, models or settings.

`results/audit.json` independently reconstructs event counts, McNemar from a binomial CDF and interaction variance from the four-cell covariance matrix. This checks arithmetic, not independent psychometric reference calibration. Actual numerical results are in `REPORT.md`, `summary.json` and `comparison.json`.
