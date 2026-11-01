# Paired ordinal DIF / Holm-FWER pilot

Base main `eb8c798` (merged #117). Preregistration commit **3d0f67b** predates
all pilot responses/results; execution implementation **da9c8df**, with saved
code/data/plan SHA256 and separate reporting-code hash. No production changes.

## Inventory, design and estimand

P2 records contain seeds and results, **not response matrices**. We regenerated
exactly its first ten ordinal-null/impact seeds 2000100–2000109 using the existing
`gen_ordinal`; old p-values were not used as paired data. The fresh matrices are
saved, with the same matrix/row order supplied to all three procedures.

Balanced N=500/500, 10 items, four ordered codes 0–3, independent GRM responses
conditional on theta; reference N(0,1), focal N(−.5,1), a=linspace(.9,1.8),
thresholds linspace(−1,1)+[−1,0,1]. **No true DIF.** Eight oracle-invariant anchors
P01–P08 are fixed in advance. The common tested family is only P09/P10; this
cost-driven scope is narrower than P2's ten-item purified screening family.

Mantel (1 df) and ordinal LR group+interaction omnibus (2 df) use the production
item functions, total of all ten items including the studied item, no purification.
GRM-LR uses production `test_dif`, joint a/all-threshold test (4 df), fixed anchors,
reference mean0/SD1 and focal mean/SD free; GH61 with GH81 audit. Alpha .05,
Holm across the same two items primary; BH on that family secondary. Components
and anchors are not extra tested hypotheses. There is no automatic anchor selection.

The **latent-invariance** and **observed-score conditional** nulls are not equivalent.
The comparison measures false-DIF operating rates under a common latent-null DGP;
it is not an equal-df/equal-power comparison of identical statistical hypotheses.

## Engineering failure and preserved evidence

The initial harness passed category labels where production GRM requires the
integer category count. All ten GRM calls were rejected **before any fit**; their
raw records/logs and summary remain in `evidence_initial_api/` (code 929e450).
Corrected only this harness contract and added a no-fit regression. Same CSV bytes,
Mantel and ordinal-LR results were reused unchanged. Exactly one actual GRM analysis
per seed then ran; no alternate starts, criterion changes or scientific fit retries.
These are ten unique paired datasets, not twenty independent repetitions.
Historical file-layout aliases can be restored exactly with `artifacts --restore`.

Final pilot: **10/10 complete valid triples**. Mantel convergence is not applicable;
ordinal LR's production gate requires all regressions to converge; all ten GRM
analyses passed the existing convergence/information/bound/quadrature gates.
The saved-data audit checks hashes, group counts, category/item alignment, equal
anchor a/b across groups, omnibus df=4 and all GRM gates. H2 max normalized score
3.49e-7; max GH61/GH81 LL discrepancy 5.88e-5. Focal H2 means −.6874..−.3758,
SD .9367..1.1014. No independent R comparison was performed or claimed.

## Results and uncertainty

Primary intervals are exact **98.333% Clopper–Pearson per method** (Bonferroni
joint coverage >=95% for three primary assessments). Failure-robust limits treat
unknown failures as events 0/1; they equal conditional limits here (zero final
failures). A method with any invalid family is never counted as a valid zero event.

| Procedure | Holm events/FWER | Primary interval | FWER plug-in MCSE | Raw Type-I | Cluster Type-I MCSE | BH-FDR (=BH-FWER under complete null) |
|---|---:|---:|---:|---:|---:|---:|
| Mantel | 1/10 = .10 | [.00084,.51524] | .0949 | .10 | .0667 | .10 |
| Ordinal LR | 1/10 = .10 | [.00084,.51524] | .0949 | .10 | .0667 | .10 |
| Fixed-anchor GRM-LR | 0/10 = .00 | [.00000,.38044] | .0000 | .05 | .0500 | .00 |

Zero observed FWER/MCSE is **not** zero population risk. Marginal 95% BH-FDR
intervals: Mantel/LR [.00253,.44502], GRM [0,.30850]. These are not simultaneous
secondary-claim intervals. No alternative/power or confidence-interval coverage
experiment was conducted. Raw Type-I refers to the two-item family, conditional
on full validity; it is not the original ten-item P2 estimand.

Paired Holm event differences use the same ten complete datasets. Mantel minus LR:
0 (MCSE0; zero discordances). Each observed-score method minus GRM: +.10 (MCSE .10;
one observed-only event, no GRM-only event). Conservative marginal paired 95%
Hoeffding intervals are [−.859,.859] and [−.759,.959], respectively. No method
superiority is established. Complete-pair counts/missing-pair sensitivity ranges
are saved; no failed pair is silently counted as concordant.

## Mechanism check without fitting or response simulation

Known-GRM total-score convolution computes E[item | total, group] by GH61/GH81.
Impact0 control gives identical curves. With impact−.5 and no latent DIF, maximum
conditional-mean contrasts are **.022584 (P09)** and **.025156 (P10)** coded-score
units. GH step discrepancies are 4.33e-8/5.31e-8; full total-score mass/mean curves
are in mechanism.json. Thus observed total score is not a group-invariant sufficient
matching variable for this GRM design. This supports the hypothesized mismatch
mechanism; it does not prove population FWER inflation from a ten-repetition pilot,
or identify all causes in other designs. No production bug was demonstrated.

## Precision, cost and scientific decision

Before fits, enumerated every possible binomial event count to find the smallest
R with worst exact-interval deviation <=.02 at confidence 1−.05/3: **R=3632**.
Worst deviation .01999845 (R3631: .02000122); worst Bernoulli MCSE .0082965.
This does not assume an event rate .05 or that 400 repeats suffice.

Measured total pilot cost ~**37.9 seconds**, conservatively including initial
API-attempt/observed-fit costs; estimated precision-planned total **13,772 s
(~3.8 hours)**. The preregistered calibration budget is **600 s**. All pilot methods
were operational, but the main experiment was **NOT EXECUTED** because projected
cost exceeds this fixed budget. No budget/tolerance was changed after seeing results.

**Mantel: INCONCLUSIVE / NOT CERTIFIED. Ordinal LR: INCONCLUSIVE / NOT CERTIFIED.
GRM-LR: INCONCLUSIVE / NOT CERTIFIED.** Pilot values do not prove control,
inflation or superiority. No production p/CI, method status or model label was enabled.
The main decision rule remains failure-robust upper<=.05 for scoped control,
lower>.05 for inflation, otherwise inconclusive, only after a precision-planned main
experiment; the diagnostic pilot cannot receive that acceptance label.

Cannot generalize to ten tested items, selected anchors/purification, unequal
sample sizes, other latent distributions, DIF alternatives, or general BH control.
Next smallest diagnostic: preregister a ten-dataset impact0 common-seed control
with unchanged procedures/family to isolate impact; still no certification. For
actual precision evidence, independently budget the planned ~3.8h study or preregister
a different precision/scope before new outcomes; do not reinterpret this pilot.

## Reproduction and verification

- `plan.json`, precision_plan.json: locked scientific design and exact required R.
- `evidence/data_*.csv`, rep_*.json: raw paired matrices, method p/statistics, full
  GRM estimates/diagnostics in methods.GRM_LR, hashes and timing. Redundant worker
  JSON and empty logs are byte-reconstructable through artifact_audit.json; see below.
- `summary_fit_run.json`: untouched execution summary; `summary.json`: cached-data
  audit/analysis, with raw-execution and reporting-code hashes distinguished.
- New targeted tests: precision sizing, same-family decisions, failures/paired
  missingness, score-convolution identity, nonzero zero-event intervals, corrected
  GRM API, and saved-data inference-gate audit. No routine CI experiment added.

From repo root, cached reanalysis only:

```
python -m validation.ordinal_dif_paired.experiment --summarize validation/ordinal_dif_paired/evidence
python -m pytest -q tests/test_ordinal_dif_paired.py
```

To reproduce the actual execution code, use commit da9c8df and the same locked plan;
run `--run NEW_OUTPUT --reuse-observed validation/ordinal_dif_paired/evidence_initial_api`.
This executes GRM fits; it is deliberately opt-in, not a CI command. Never overwrite
archived evidence or silently refit cached datasets. Final targeted/CI status and
commit are recorded in the PR. Main is not merged by this task.


## Impact=0 control follow-up (same ten seeds; no negative-condition refits)

Control plan committed at **1cc8a85**, before control outcomes. Only focal mean
changes to 0; N500/500, J10/C4, true DIF absent, P01–P08 anchors, P09/P10 family,
matching, alpha, native fitting settings and all validity gates remain identical.
Exactly ten new control datasets were analyzed, ~**32.3 seconds** total. All three
methods have 10/10 complete valid results; no failed new analyses. Original
negative-condition CSVs, repetition records and summaries remain byte-identical.

| Impact | Procedure | Holm-FWER | Primary interval (98.333%) | FWER MCSE | BH-FDR | Raw Type-I (cluster MCSE) |
|---|---|---:|---:|---:|---:|---:|
| −.5, cached | Mantel | .10 | [.00084,.51524] | .0949 | .10 | .10 (.0667) |
| 0 | Mantel | .10 | [.00084,.51524] | .0949 | .10 | .10 (.1000) |
| −.5, cached | Ordinal LR | .10 | [.00084,.51524] | .0949 | .10 | .10 (.0667) |
| 0 | Ordinal LR | .10 | [.00084,.51524] | .0949 | .10 | .05 (.0500) |
| −.5, cached | GRM-LR | .00 | [0,.38044] | .0000 | .00 | .05 (.0500) |
| 0 | GRM-LR | .00 | [0,.38044] | .0000 | .00 | .00 (.0000) |

BH-FDR equals BH-FWER under these complete nulls. Its marginal 95% binomial
intervals are [.00253,.44502] for the observed-score methods and [0,.30850] for
GRM, in both conditions. Primary confidence coverage is >=95% across three
methods **within each condition**, not across all six condition/method cells.

### Verified common randomness, not an assumed statistical gain

Seed equality alone was not used as proof. The actual impact0 generator was
instrumented: two standard-normal vectors (reference/focal) and ten item-uniform
vectors were traced with call names, shapes, SHA256, BitGenerator and NumPy version.
The generator source was checked against 5da63fa. With those actual control draws,
applying focal shift −.5 reconstructs every stored negative response **exactly**;
reference responses are identical and focal responses are monotonically coupled.
No negative-condition model or p-value was recalculated. This verifies the intended
common-random-number structure under the recorded source/runtime. It cannot verify
unarchived older P2 bitstreams and is not a guarantee of strong outcome correlation.

Across-impact Holm and BH paired differences (impact0 minus −.5) are **0** for all
three methods, empirical MCSE0, ten complete pairs, marginal conservative 95%
interval **[−.85894,.85894]**. Thus no zero impact effect or equality is established.
Observed-score event vectors are identical (empirical correlation ~1 on ten
observations); GRM event correlation is undefined because both vectors are zero.
Neither fact estimates population pairing strength reliably. Raw Type-I differences
are 0 (Mantel, MCSE .07454), −.05 (LR and GRM, MCSE .05); their broad marginal
intervals also include zero. Within either condition, observed-score minus GRM
Holm difference is +.10 (MCSE .10, interval [−.75894,.95894]); no superiority follows.
Full event/difference vectors, empirical correlations, failures and missing-pair
sensitivity ranges are in evidence_control/comparison.json and raw rep_*.json.

### Audit of the original 112-file change

At 5da63fa: **112 files, 1,379,181 bytes**, largest 42,463 bytes; no oversized
transient model data. Removed **72** losslessly reconstructable files: 40 empty
logs, 20 redundant worker JSON files already embedded in canonical repetition
records, two duplicate initial precision/mechanism files, and ten byte-identical raw CSV aliases. Every removed byte
stream has its original path, size, SHA256 and reconstruction recipe in
artifact_audit.json. The full 112-file inventory is retained. One canonical copy of each negative-impact matrix remains; `--restore` reconstructs
byte-identical CSV aliases for historical harness compatibility. Canonical raw responses, fits,
initial API errors and original summaries were not removed or recomputed.

```
python -m validation.ordinal_dif_paired.artifacts          # verify 72 recipes, no writes
python -m validation.ordinal_dif_paired.artifacts --restore  # optional exact old layout
python -m validation.ordinal_dif_paired.control --summarize validation/ordinal_dif_paired/evidence_control
```

The opt-in `control --run NEW_OUTPUT` executes only the ten impact0 controls, no
expansion. It is not a CI command. Control records distinguish control-driver,
unchanged worker, generator and plan hashes. Four small follow-up tests cover RNG
coupling, actual paired uncertainty, all reconstruction hashes and saved-control
inference/anchor audits; no fits occur in these tests.

**Decision unchanged: INCONCLUSIVE / NOT CERTIFIED for all three methods.** This
control does not demonstrate FWER control, general impact invariance, or superiority.
The analytical matching-null mismatch remains a candidate mechanism; its effect
on operating error rates is not established by ten paired trials. No model status,
production inference gate or scientific threshold changed. No broad MC study,
full regression or R calibration was launched for this follow-up. Normal mandatory
CI status is recorded in PR #118; no merge.
