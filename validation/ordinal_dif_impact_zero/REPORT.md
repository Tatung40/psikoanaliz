# Impact0 control of paired ordinal DIF — scoped results, not certification

## Execution and pairing

Base main `538e974` (merged PR120). Plan `88ba781` committed before new outcomes; executable `7f32a58` committed before fitting. No production estimator, p/CI gate, scientific label or old evidence changed. No R, GRM-LR or 4PL calibration or local full regression.

Same design as PR120 except **both theta distributions N(0,1)**: N500/500, J10, four categories, invariant latent GRM, local independence; P09/P10 common family, production ordinal-logit omnibus LR, total/rest conditioning, Holm .05, secondary BH. No purification/selected anchors. Exact same2500 seeds and order (`2026110100`–`2026112599`).

Feasibility **30/30 valid**, 12.6718 seconds, projected2500 cost1055.98≤1800 seconds; continuation based only on validity/runtime. Those30 counted toward fixed2500. Final **2500/2500 valid pairs**, **0 failed/replaced/retried**, **1044.4647 seconds (17.41 minutes)**. Every new item/method LR passed the unchanged production convergence gate.

Common randomness is **verified, not inferred from equal seeds**: source hashes and all four package versions match PR120. For every seed, actual control generator calls were traced (two normal arrays, ten item-uniform arrays). Reconstructing old responses using these same draws with only focal theta shifted−.5 reproduced **all2500 archived response/group hashes exactly**. Reference rows are identical; focal changes obey the shared-uniform monotonicity check. Draw hashes/layout are preserved. This reconstruction performs **no old fit**. A manifest verified **all39 PR120 files unchanged** before and after execution.

## Method-specific estimates

| Impact | Method | Holm events/2500 | FWER | 97.5% CP interval | FWER MCSE | BH global-null FDR | Raw itemwise false-positive rate |
|---|---|---:|---:|---|---:|---:|---:|
|0|total|111|.0444|[.035648,.054532]|.004120|.0448|.0462|
|0|rest|124|.0496|[.040341,.060221]|.004343|.0500|.0492|
|−.5 (cached)|total|171|.0684|[.057531,.080570]|.005050|.0692|.0704|
|−.5 (cached)|rest|533|.2132|[.195077,.232180]|.008193|.2164|.1784|

Impact0 BH97.5% CP intervals: total [.036008,.054971], rest [.040704,.060657]. Raw-rate repeat-clustered MCSE: .002951/.003059. Under the global latent null, BH FDR equals the probability of any BH rejection; no mixed-null/power claim follows. Method CP intervals have joint≥95% **within each condition**, not jointly across all four cells and contrast/BH families. Old estimates/intervals above are reused, not recalibrated.

## Paired contrasts

All contrasts use2500 complete pairs. b=positive-condition-only event, c=negative-condition-only event; a=both, d=neither. Signs below are fixed in the preregistration.

| Contrast | a,b,c,d | Difference | MCSE | Marginal conservative exact95% CI | Exact McNemar p | Holm p (3 tests) |
|---|---|---:|---:|---|---:|---:|
|Primary T0−R0|91,20,33,2356|−.0052|.002911|[−.015321,.002305]|.0983706|.0983706|
|Total T0−Tm|56,55,115,2274|−.0240|.005194|[−.040895,−.010273]|4.87027e−6|9.74053e−6|
|Rest R0−Rm|78,46,455,1921|−.1636|.008336|[−.190502,−.137087]|1.14119e−85|3.42358e−85|

m=impact−.5. The interaction **(T0−R0)−(Tm−Rm)=+.1396**, paired MCSE **.00788350**. Supplementary conservative exact paired95% CI **[.102805,.173213]**. All16 quadruple cells and3×3 signed-event cells are preserved in `comparison.json`; **McNemar is not applicable to a nonbinary difference of differences**.

| Contrast | Marginal95% Hoeffding CI | Simultaneous four-contrast Hoeffding CI |
|---|---|---|
|Primary T0−R0|[−.059524,.049124]|[−.068919,.058519]|
|Total impact change|[−.078324,.030324]|[−.087719,.039719]|
|Rest impact change|[−.217924,−.109276]|[−.227319,−.099881]|
|Interaction|[.030952,.248248]|[.012162,.267038]|

The last column uses Bonferroni alpha=.05/4: **joint coverage≥95% over all FOUR contrasts**. Other95% intervals are marginal, not jointly95%. Exact McNemar3-test Holm correction is a separate prespecified family. McNemar **never certifies nominal FWER**. Hoeffding intervals are conservative/wide; no favorable interval was selected after observing results. R2500 bounds the binary-difference MCSE by .02; interaction range[−2,2] instead has worst-case MCSE bound .04, with actual MCSE reported above.

## Scientific decisions A–D

- **A:** impact0 point estimates are near nominal and both CP intervals include .05. **Control/equivalence is not certified**: upper bounds exceed .05, and no equivalence margin was declared.
- **B:** no evidence of impact0 inflation under the preregistered lower-CP>.05 rule. This is not proof of exact nominal control.
- **Primary:** no total/rest directional difference established at impact0; neither equality nor equivalence is proved.
- **C:** rest-score impact change is supported even by the simultaneous Hoeffding rule. Total's change is supported by supplementary exact paired CI/Holm-McNemar, **but its simultaneous Hoeffding interval includes0**, so the preregistered joint cross-contrast rule does not confirm its direction.
- **D:** positive interaction is supported by the simultaneous interval: changing the focal mean from−.5 to0 reduces rest's operational false-DIF risk more than total's, by13.96 percentage points here. This is a paired interaction, not a comparison of independent MCSEs.

Causal interpretation is restricted to the controlled focal latent-mean change in this synthetic DGP; CRN is a simulation coupling. Total/rest condition on different observed variables. Latent item invariance does not guarantee either score-conditional null, nor do these results demonstrate a production optimizer defect. No certification/generalization to other item families, group sizes, impact, lengths, categories, selection or alternative engines.

## Audit, storage and tests

Independent arithmetic audit:5000 new method-level decisions checked directly from raw p values using two-item Holm/BH formulae; b/c and exact McNemar reconstructed with a binomial CDF; interaction MCSE independently computed as sqrt(c'Cov(T0,R0,Tm,Rm)c/n), c=(1,−1,−1,1). It equals .007883501496057108 versus direct-difference .007883501496057151 (floating roundoff). This is an arithmetic audit, not independent R calibration/scientific certification.

All2500 raw records, CRN traces, metadata/source/version/hash records, first30 matrices, feasibility, comparison, audit and execution log are committed (2.8MB). Remaining data are deterministic from seed/source/version. No failures occurred; empirical missing-outcome envelopes collapse to observed estimates, not zero confidence uncertainty.

**7 distinct targeted tests passed**: five before calibration, one independent-audit fixture, one post-run evidence integrity/pairing/recalculation test. No successful test was repeatedly run locally. Normal PR CI includes the lightweight test file; it does not launch this experiment. Live CI status is on PR121 and is not scientific certification. Do not merge automatically.
