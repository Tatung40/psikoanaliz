# Fixed-anchor GRM-LR paired feasibility — NOT CERTIFIED

Base main `2b1f619` includes PR121. Preregistration `bc65e4e` precedes new fits; executable code was committed before fitting (revision/source hashes in metadata). Only this feasibility was executed. No production algorithm, inference gate, scientific label or previous ordinal LR analysis was changed/repeated.

## Execution and cost

**10 seed pairs ×2 impact conditions =20 GRM-LR dataset analyses. All20 were valid and converged;0 failures/timeouts/external retries.** Production caching was retained:140 optimizer calls,160 item×hypothesis fit slots. Seeds2026110100–2026110109, exact first10 from the original2500 family.

| Condition | Analyses/valid/converged | Worker wall total | Mean per analysis | Projected2500 analyses |
|---|---|---:|---:|---:|
|impact−.5|10/10/10|26.3683s|2.63683s|6592.07s (1.83h)|
|impact0|10/10/10|28.2318s|2.82318s|7057.94s (1.96h)|

Total end-to-end **55.1849s**, mean **2.75924s/analysis**, below600s. Peak worker RSS **187.1719MiB** (Linux ru_maxrss), below1024MiB safety threshold. Full2500 seed pairs require **5000 GRM analyses**, projected **13796.22s =3.83h**, including observed orchestration overhead. This is a ten-seed runtime extrapolation, not a guaranteed bound. **The main experiment does not fit the current600s budget and was not started.**

## Scientific/API gates

Unchanged production `core.grm_dif.test_dif`, via PR118's existing wrapper. Categories are **integer4**, GH61 and same-estimate GH81 audit; no additional quadrature calibration. Same prior API/settings verified by a targeted contract test, not by reusing old PR118 results as new paired evidence.

N500/500, J10, four ordered categories, invariant latent GRM DGP, no missingness. P01–P08 oracle invariant anchors; P09/P10 common family. Reference N(0,1), focal mean/SD freely estimated with native bounds. Anchors share slope and all3 ordered thresholds, fixing intergroup scale; positive slopes fix orientation. Oracle invariance is known from the DGP and enforced, **not empirically proved by an anchor test**. Joint target slope/threshold LR has4df: full50 vs restricted46 parameters; threshold-only and slope-only components are not new testing families in this study.

All160 hypothesis slots passed native convergence, transformed-bound, quadrature and observed-information gates, plus exact anchor-equality and physical-feasibility audits:

- Maximum per-observation score **5.60375e−7** (<2e−6).
- Maximum GH81 audit score **2.50627e−6** (≤1e−5).
- Maximum GH61/GH81 logLik difference **.000111583** (≤.001).
- Minimum information eigenvalue **6.84714**; maximum condition number **155.494**; all native relative positive-definiteness criteria and covariance-availability checks passed.
- **0 boundary flags**;160 exact anchor-equality audits passed. Native slope/ordered-threshold/latent bounds and optimizer settings were preserved.
- Raw LR from saved logLik,4df and p values were checked; Holm/BH were recomputed on exactly P09/P10.

These are numerical feasibility checks, **not independent R calibration, coverage validation or global identification certification**.

## Pilot events and uncertainty

Each condition observed **0/10 Holm events and0/10 BH events**. Individual97.5% CP interval **[0,.354805]**; preregistered nine-component joint-family CP interval **[0,.444902]**. Empirical FWER MCSE=0 does **not** imply zero uncertainty or nominal control. Raw itemwise false-positive rates were .10 (impact−.5) and .05 (impact0); cluster MCSE .06667/.05. These10-repeat counts do not establish superiority, control, power or general validity.

`summary.json` contains four within-condition GRM−ordinal contrasts, GRM's paired impact contrast, two method×impact interactions, all paired discordances/quadruple tables, MCSE, exact paired intervals, Hoeffding intervals, and five raw/Holm McNemar p values. No McNemar is applied to nonbinary interactions; it never certifies nominal FWER. All are **pilot descriptions only**.

The predeclared joint family contains2 GRM FWER intervals plus7 contrast intervals, each alpha=.05/9 (CP/Hoeffding), yielding joint≥95% over those9. Supplementary paired95% intervals and97.5% CP intervals do not acquire that global coverage. The formal GRM latent-null and ordinal LR observed-score conditional null differ: comparisons measure operational false-DIF decisions against the same latent-null DGP, not interchangeable statistical hypotheses.

## Pairing, preservation and next step

All20 response/group hashes equal their archived PR120/121 counterparts. Common normals/uniforms are checked through the previously verified control reconstruction; reference rows/group order agree. Both methods' ordinal outcomes are read from their original raw archives, **never refitted**. All81 canonical tracked files are hash-verified unchanged. Temporary ignored logs are not manifest inputs; canonical execution text and worker stdout/stderr are preserved in the new archive.

Committed artifacts:20 compact matrices, all raw native fit diagnostics/physical parameters/LR/p and decisions, worker logs, timing/RSS, plan/source/version/hash metadata, checksums and summary. No failed records were silently dropped; no failures occurred. **5 distinct targeted tests passed**: anchor/df/identification, failure gates, canonical plan protection, existing API/category contract, then saved-evidence integrity and Holm/BH recalculation. No local full regression/new R fits were run. Existing GitHub CI runs the lightweight tests, not this experiment.

Next step is a **separate budgeted stage**, not an automatic main run. Proposed immutable cumulative stages100→500→2500 seed pairs reuse the completed10, with additional projected costs497s→2207s→11037s. `next_stage_plan.json` proposes a cumulative18000s resource cap, stage limits,90s per-analysis timeout and memory/validity stops. Budget headroom is not statistical precision. Source/settings changes prohibit automatic pooling; failures remain without replacement; incomplete or invalid fixed-R results remain inconclusive. **No scientific certification or production-label change is justified by this feasibility. Do not merge automatically.**
