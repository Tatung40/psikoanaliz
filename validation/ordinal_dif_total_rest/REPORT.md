# Ordinal LR total/rest paired experiment — completed, scoped evidence

**Decision:** in this fixed latent-null/impact−0.5 design, rest-score conditioning increases operational false-DIF Holm-FWER. Both methods show scoped inflation; neither is certified for nominal .05 control here. Production algorithms, inference gates and labels are unchanged. No GRM-LR, R or 4PL calibration was run.

## Preregistration and execution

Augmented plan committed **before new data/outcomes** in `30bf6a3`; tested harness committed before fitting (exact revision and source hashes in `results/metadata.json`). Base main `b030cd7`, merged #119. No post-outcome threshold or design modification.

N=500 reference +500 focal, theta N(0,1)/N(−0.5,1), J=10, C=4, invariant GRM item parameters, local independence; same generated matrix/group/order for total and rest. Tested family P09/P10, production ordinal-logit omnibus LR, Holm alpha=.05; BH descriptive secondary. Total includes the studied response; rest subtracts it. No purification, selected anchors or alternative starting values. Seeds `2026110100` through `2026112599`.

First30: **30/30 valid**, 13.5066 s, projected2500 cost1125.55 s≤1800 s. Only validity/runtime selected continuation. These30 are included in the fixed total. Final: **2500/2500 complete valid pairs**, **0 failures/replacements/retries**, **1083.4756 s (18.06 min)**, within budget. No outcome-based early stop. All four item/method LR analyses per dataset passed the production convergence gate. There is no independent-reference certification claim.

## Method-specific outcomes

| Measure | Total score | Rest score |
|---|---:|---:|
| Holm events /2500 |171|533|
| Holm-FWER |.0684|.2132|
| 97.5% exact CP CI |[.057531, .080570]|[.195077, .232180]|
| Repeat-level FWER MCSE |.005050|.008193|
| BH global-null FDR |.0692|.2164|
| 97.5% exact CP BH CI |[.058269, .081429]|[.198172, .235475]|
| Raw itemwise false-rejection rate |.0704|.1784|
| Raw rate repeat-clustered MCSE |.003558|.005374|

The two Holm intervals have joint coverage at least95% (Bonferroni). Both lower bounds exceed .05: scoped inflation is supported, rather than an argument based only on point estimates. BH intervals form a separate secondary two-method family; no simultaneous95% claim covers all Holm and BH intervals together. Under this global latent null, BH FDR equals probability of any BH rejection; it is not a mixed-null/power/selection result.

## Primary paired comparison

D=Holm event(total)−Holm event(rest). Both events a=154; **b=17 total-only**, **c=379 rest-only**, neither d=1950.

- Δ=**−.1448** (rest higher by14.48 percentage points).
- Repeat-clustered paired MCSE=**.00741587**, below prespecified .02 target; worst-case bound1/√2500=.02.
- **Primary95% Hoeffding CI [−.199124, −.090476]** excludes zero, satisfying the preregistered directional rule. This conservative, distribution-free interval is deliberately wide; .02 was an MCSE target, not a95% half-width target.
- Supplementary conservative exact-coverage paired95% CI **[−.167318, −.121942]**, using the prespecified CP rectangle for discordance probability and conditional direction.
- Exact two-sided McNemar **p=3.7278025×10⁻⁹⁰** (396 discordances). This tests equal marginal event probabilities; it **does not certify nominal FWER control**. It was not used to override the primary rule.

There were no failed pairs, so unknown-failure sensitivity bounds collapse to the observed empirical rates/difference. Confidence uncertainty remains nonzero. Point estimates, MCSE and intervals must be read together.

## Interpretation and limits

Latent item invariance does not imply either observed-score conditional null. Total and rest match on different variables, and this design measures their operational false-DIF behavior against the latent-null DGP. It does not prove a defect in the ordinal-logit optimizer or establish universal superiority of total matching. Rest exclusion does not remove the impact-induced posterior-ability mismatch; prior #119's analytic findings remain consistent with these observations.

Evidence is restricted to this DGP, impact, N/J/categories, P09/P10 family and no-selection setting. It does not establish impact0 behavior, general type-I control, power, mixed-null FDR, alternate anchors, other lengths/discriminations or other engines. No production p/CI gate is opened. Next separate preregistered extension could test impact0 with the same family and precision policy; no such new computation was launched here.

## Reproducibility and checks

`plan.json`, `experiment.py`, metadata and source hashes preserve settings; `results/records.jsonl.gz` preserves all2500 raw repeat records (LRs, p values, adjustments, validity, seeds, content hashes and times). First30 matrices are compressed NPZ; remaining response matrices are deterministically reproducible from the original generator/version/seed. No historical files were overwritten. `results/summary.json`, feasibility record and execution log preserve actual outcomes.

Targeted verification: five deterministic harness tests before fitting, then one stored-evidence test after completion (**6 distinct tests passed**). The latter independently rebuilds summaries from raw records, checks all stored p-family decisions, unique seeds, matrices/content hashes and source hashes. No full regression or new calibration was run locally. Existing PR CI provides integration checks; live status is on the PR, not scientific certification.
