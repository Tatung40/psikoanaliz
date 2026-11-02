# Staged GRM-LR continuation: technical stop, INCOMPLETE / NOT CERTIFIED

**Decision:** stages100 and500 completed; stage2500 stopped at the first quadrature-score failure as preregistered. No failed fit was retried/replaced, no threshold changed, no later seed selected. Neither nominal FWER control nor method superiority is certified.

## Design, execution and immutable reuse

Base main `0bd4eb6` includes PR122. Preregistration `b448031` was committed before any new fits; final pre-fit plan and executable revision/source hashes are recorded in `results/metadata.json`. Original20 pilot fits are loaded unchanged; **1743 new attempts**, **1763 total attempts**, **1762 valid**, **1 invalid**. All1763 dataset analyses met native optimizer convergence; one failed numerical regularity. There were0 timeouts and0 external retries. **881 complete valid seed pairs**; the882nd seed's negative-impact analysis failed and its impact0 counterpart was not attempted.

Same N500/500, J10, four categories, invariant latent GRM, anchorsP01–P08, tested familyP09/P10, reference N(0,1), focal mean/SD free,4df omnibus LR, Holm .05, secondary BH; unchanged GH61/GH81, bounds, information and optimizer gates. Existing ordinal LR outcomes are read from archives, **never refitted**. Data/content hashes and common-normal/uniform replay agree with prior records; all111 protected canonical files remain unchanged.

| Technical stage | Complete pairs | Analyses / valid | Cumulative seconds including pilot |
|---|---:|---:|---:|
|A|100|200/200|551.175|
|B|500|1000/1000|2740.476|
|C, incomplete|881 complete +1 failed negative-only attempt|1763/1762|4839.208|

Total **80.65 minutes**, including reused pilot55.185s; new execution4784.023s. Mean2.74487s/attempt; peak worker RSS **210.828MiB**. Neither18000s cumulative budget, stage cap,90s/analysis timeout nor1024MiB memory threshold caused stopping. Budget remaining was not permission to override the scientific gate. Planned but unattempted:1618 negative-impact and1619 zero-impact analyses.

## Exact stop evidence

Seed **2026110981**, index881, impact−.5. All8 hypothesis slots converged, no boundaries, anchors equal and information matrices usable. GH61→GH81 **audit gradient**, not optimizer convergence, was unacceptable:

| Slot | Maximum per-observation GH81 score | GH61/GH81 logLik difference |
|---|---:|---:|
|P09 Ha|1.08520908e−5|.000415866|
|P09/P10 H2 shared solution|1.19966271e−5|.000457361|
|P10 H0|**1.32178385e−5**|.000500691|
|P10 Hb|1.23454715e−5|.000490862|
|P10 Ha|1.26170984e−5|.000461599|

Native audit-score limit **1e−5** exceeded; all listed logLik differences still satisfied .001. Production suppressed both item p values; the common2-item family was invalidated. Failed matrix and all estimates/diagnostics/logs are retained. No GH81 refit, denser-grid calibration, tolerance change or optimizer repair was attempted. This demonstrates a numerical integration limitation for this saved solution, not structural nonidentifiability or an optimizer failure.

## Conditional descriptions — NOT confirmatory results

The planned fixed2500 design is incomplete and stopping depended on numerical validity. The following **valid-case prefix** estimates and fixed-sample interval formulae are descriptive. **The preregistered joint95% coverage guarantee and confirmatory decisions are not asserted after this technical early stop/selection.** No sequential confidence method was preregistered or substituted post hoc.

| Impact | Valid / attempted | Holm events | FWER | MCSE | Nine-family CP formula interval | BH global-null FDR |
|---|---|---:|---:|---:|---|---:|
|−.5 primary|881/882|57|.064699|.008292|[.043901,.091054]|.066969|
|0 secondary|881/881|55|.062429|.008156|[.042023,.088424]|.064699|

Both intervals contain .05; even their fixed-sample formulae do not establish inflation or control. Individual97.5% CP formula intervals: minus [.047427,.085749], zero [.045473,.083186]. For the882 negative attempts, assigning the single invalid event either outcome yields empirical risk bounds [.064626,.065760]; it is not silently assigned “no rejection.” Unattempted cases remain unknown, not zero events. Raw itemwise rates .065834/.055051 with repeat-clustered MCSE .006029/.005744. BH event probability is FDR only under this global latent null.

Per-item decisions among881 valid analyses/condition:

| Impact | Item | Raw / Holm / BH rejections |
|---|---|---|
|−.5|P09|55 /33 /35|
|−.5|P10|61 /28 /30|
|0|P09|42 /28 /30|
|0|P10|55 /33 /35|

## Paired conditional comparisons

All contrasts below use **the same881 complete valid seeds**, including cached ordinal decisions for that prefix. The prior2500-replicate ordinal FWER values are **not** substituted into these paired differences. GRM−ordinal signs are fixed; impact change is zero-minus-negative.

| Contrast | Difference | MCSE | b/c | Nine-family Hoeffding formula interval | Raw McNemar p / Holm5 p |
|---|---:|---:|---|---|---|
|GRM−total, minus|−.002270|.008942|30/32|[−.117866,.113325]|.899076 /1|
|GRM−rest, minus|−.145289|.014606|28/156|[−.260885,−.029694]|9.693e−23 /4.846e−22|
|GRM−total, zero|.014756|.007602|29/16|[−.100840,.130351]|.072454 /.289817|
|GRM−rest, zero|.010216|.007611|27/18|[−.105380,.125811]|.232693 /.698080|
|GRM zero−minus|−.002270|.009901|37/39|[−.117866,.113325]|.908777 /1|
|Interaction GRM−total|.017026|.010819|N/A|[−.214165,.248217]|N/A|
|Interaction GRM−rest|.155505|.015560|N/A|[−.075686,.386696]|N/A|

Exact paired interval formulae and all16-cell interaction tables are preserved in `summary.json`; these are also descriptive after stopping. The negative-impact rest comparison is conspicuous in the observed prefix, **not a completed-study superiority finding**. McNemar compares marginal binary event probabilities, does not certify nominal FWER, and is not applied to nonbinary interactions. GRM latent invariance and observed-score conditional ordinal LR test different null hypotheses; comparisons measure operational false-DIF behavior against this DGP.

At fixed complete design the family was2 CP FWER intervals plus7 Hoeffding contrasts, each alpha=.05/9; five binary McNemar p values receive separate Holm correction. No family, alpha, hypothesis or acceptance limit changed after outcomes.

## Independent audit, tests and reproducibility

Independent code reconstructed per-item Holm/BH from raw p values; dataset FWER/FDR, raw rates and clustered MCSE; paired b/c, differences and binomial-CDF exact McNemar; CP endpoints by **binomial-tail root inversion** independently of beta quantiles; exact paired intervals and joint-family Hoeffding formulae. Interaction MCSE was checked through the four-event covariance matrix. Independent Counter calculations matched all32 cells across the two interaction tables. Archive and source SHA256 checks and111-file protection passed.

Original20 fits remain references to the immutable pilot. New raw records/logs are deterministic gzip chunks of≤100 analyses, with checksums, append-only durable ledger during execution, source/version/seed/data hashes, stage checkpoints and failed matrix. Completed A/B archives were backed up on the same branch without new CI runs. No original evidence was rewritten. **5 distinct targeted tests passed**; a pre-fit harness path error was corrected before new calibrations, with only its failed test repeated. No local full regression/R calibration/old ordinal refit was run.

**Next smallest step:** a separate, preregistered no-refit quadrature diagnostic on the saved failed solution before proposing any denser-grid study. This PR does not change production or inference gates and does not restart the interrupted design. Status remains **INCOMPLETE / NOT CERTIFIED**. Existing GitHub CI verifies cached evidence/lightweight tests; it does not run this Monte Carlo. Do not merge automatically.
