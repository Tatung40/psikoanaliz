# Uniform cold-start GH101/GH161 pilot — TECHNICAL PASS / NOT CERTIFIED

**Decision: PASS for the preregistered technical pilot only.** All20 predefined datasets (10 paired seed units) completed;140/140 unique hypotheses converged and passed numerical gates. No failures, substitutions, retries or density escalation. **100/500/2500 stages were not started.** This is not nominal FWER control, statistical equivalence, power evidence or scientific certification.

## Protocol, sample and actual cold starts

Protocol `GRM-GH101-COLD-PILOT-1`, commit `649acaf`, was committed before new pilot outcomes. Pre-fit executable revision and source SHA256 are in metadata. Base PR125 commit `5dda968`. Fixed seeds **2026110100–2026110109**, both impacts−.5/0; these are the first10 originally prescribed seeds, distinct from single-case2026110981. Reuse of archived data is explicit: no new response matrices were simulated.

Each dataset used N500/500,J10,C4, anchorsP01–P08, targetsP09/P10; unchanged referenceN(0,1), focal mean/SD free, positive slopes, ordered thresholds, native bounds and4/3/1df hypotheses. GH101 fitting and independently constructed SciPy GH161 audit are uniform. Every unique hypothesis starts with **start_fit=None**: native pooled-tail thresholds, all slopes1, focal mean0/SD1. No GH61 or previous GH101 parameters, and no within-dataset warm-start hierarchy, were used. Fresh dataset fitters/cache plus7 unique role layouts prevent cross-dataset/hypothesis reuse; shared H2 is evaluated once for8 reported slots.

Actual optimizer x0 was transparently recorded and independently reconstructed from category-frequency counts. **140/140 starts matched exactly**,0 initial clipping,140 SciPy successful terminations, and140 native optimizer calls. Native maxiter2000/ftol1e−14/gtol1e−10/bounds are unchanged. Counts of140 fits/160 slots are not140/160 independent datasets; the independent replication unit is the10 seeds, with paired impact conditions.

## Validity, timing and memory

| Impact | Attempted / valid / native-converged datasets | Total case seconds |
|---|---|---:|
|−.5|10 /10 /10|128.2072|
|0|10 /10 /10|130.0476|

Main pilot total **258.5328s**, mean **12.9266s/dataset**; cases range **11.9106–14.8421s**. Native fitting/information calls sum **56.6463s**; remaining time includes every-hypothesis dense Hessian, independent full-vector finite differences, logLik comparisons and durable evidence writes. Independent post-run audit takes **1.5972s** separately. No wall cap (600s total/45s per dataset) triggered.

Process RSS high-water recorded through the analysis phase **145.2656MiB**, below1024MiB. This is a cumulative single-process high-water metric, not20 independent per-case memory peaks. Convergence/numerical validity rates are20/20 (100%) descriptively; this small pilot does not prove a population success probability of100%.

| Gate | Worst observed | Preserved criterion |
|---|---:|---|
|Native GH101 normalized score|3.78034726e-07|<2e−6|
|Independent GH161 normalized score|3.78039538e-07|≤1e−5|
|Absolute GH101/GH161 logLik difference|9.98330307e-08|≤1e−3|
|Native information minimum eigenvalue|6.84713937|native relative-PD rule|
|Dense GH161 information minimum eigenvalue|6.84713937|same relative-PD rule|
|Dense information maximum condition|155.493235|reported; eigenvalue rule governs|
|Maximum LR fit/audit difference|1.16073352e-07|native nested checks; implied per-fit error bound .004|

Worst GH161 score occurs at seed **2026110104**, impact **zero**, slot **P09.H2**. All140 solutions have exact anchor equality, no active-bound flags, positive ordered gaps and available native covariance. Native and dense information satisfy minEigen>1e−7×max(maxEigen,1). The1e−5 validity limit and production p-value suppression were not changed. No new p values/SE/CI or application status labels were produced.

## Independent audit and pairing

All20 NPZ file/content/group hashes match original records. For each seed the archived two normal and ten uniform arrays were independently replayed and matched by SHA256 (120 draw hashes total), without simulating response data. Reference responses/group ordering agree exactly and focal responses satisfy the archived common-uniform monotonicity. Seed equality alone was not used as proof of pairing. All159 protected canonical files are unchanged.

Independent audit verified140 optimizer starts,140 completed fits,20 valid datasets,240 LR grid records and280 direct-row likelihood evaluations. Max native/direct-row logLik discrepancy **5.45696821e-12**. All6680 free-coordinate scores were checked at GH161 using two central-difference steps; max errors **3.71555143e-09** (h1e−4) and **2.58161974e-10** (h1e−5). These are inherited arithmetic checks, not a changed scientific score tolerance. Dense information was computed at GH161 for every hypothesis, not inferred from fitting-grid convergence alone.

## Paired GH61 diagnostics only

Cached legacy GH61 estimates were evaluated on the SAME GH161 grid, never recalibrated. Their native within-dataset warm-start hierarchy differs from this strict cold policy, so estimate/LR differences cannot be attributed to quadrature alone. No estimates, p values, FWER/FDR counts or failure labels were merged into the old Monte Carlo.

Max physical parameter difference versus legacy **1.00352639e-05**; max same-GH161 LR difference **4.12910595e-09**. New-minus-old common-grid logLik changes range **-3.72892828e-10 to 6.85759005e-09**, including small negative numerical differences, not cherry-picked improvements. Legacy nested-LR diagnostic notes:0. Full paired per-hypothesis/per-item LR decompositions are retained in raw records.

These ten old GH61 cases were already numerically well behaved; agreement here does not contradict the independently identified failure at seed2026110981, prove GH61 adequate generally, or establish cold/warm global-optimum equivalence. The single-case GH101 reoptimization from PR125 is not included as an extra pilot observation.

## Reproducibility, tests and next boundary

`records.jsonl.gz` contains every attempted dataset, actual optimizer starts/status/options, all solutions/gradients/information/FD checks, legacy comparisons, timing and raw LR records. Independent audit and protocol/data/source/version metadata are separate JSON artifacts; execution text is canonical, not an ignored temporary log. Remaining interpretation is **TECHNICAL PILOT ONLY / NOT CERTIFIED**.

Canonical raw SHA256: `c52f69bb2e525aae69f07f723f6ec495a00d23028cd8104ca7f8007cec590e91`. Gzip SHA256: `6d9b12c973f57590a1257500d6a67e918458e5c1003880d6d0010760962fb83a`. Round-trip compression integrity was checked. Input/source SHA256 lists and frozen base revisions permit deterministic data/estimation reproduction; timing and RSS are observations, not expected bit-identical rerun values.

**4 distinct targeted tests passed**: fixed seeds/data/RNG, actual native cold-start call interception with optimizer forbidden in the test, complete/unique/all-gate decision rules, and cached evidence plus independent audit. CI tests use fixtures/cached records only, not this pilot or a calibration. No local full regression was launched. Existing mandatory CI is retained; its green status is not scientific certification.

The next action is review of this pilot and a separate prespecified resource/statistical protocol before any100/500/2500 expansion. Do not simply extrapolate full-pilot runtime to an optimization-only workflow: the all-coordinate FD/dense-information audit overhead is explicit. No larger experiment or production modification was executed. Do not merge automatically.
