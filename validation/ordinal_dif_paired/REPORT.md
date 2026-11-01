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
- `evidence/data_*.csv`, rep_*.json, grm_*.json/stdout/stderr: raw paired matrices,
  method p/statistics, full GRM estimates and diagnostics, hashes and timing.
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
