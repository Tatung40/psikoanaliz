# Preregistered paired total/rest-score ordinal LR experiment

This opt-in validation experiment changes no production estimator, inference gate or scientific label. The plan was committed before generating new outcomes (`30bf6a3`), and the executable/tested harness was committed before fitting. Base: merged PR #119, main `b030cd7`.

Run once from repository root, using the existing project environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 python -m validation.ordinal_dif_total_rest.experiment
```

The harness refuses to overwrite prior evidence. Routine CI only tests small deterministic examples and stored evidence; it never launches the experiment. No GRM-LR, R calibration or 4PL fit is performed.

## Fixed scope and definitions

N=500/500, J=10, four ordered categories, impact −0.5, invariant latent GRM parameters and local independence. Only P09/P10 form the common Holm/BH family (alpha .05). Each generated response matrix, group vector and row order are shared by both methods. Total includes the studied item; rest subtracts it. Each method calls the unchanged production ordinal-logit omnibus LR implementation (two regression terms, group and score×group). There are no estimated anchors or purification.

These two matching variables define different conditional null hypotheses. “Type I error/FWER” here means operational false-DIF risk under latent item invariance; it does not establish that either observed-score conditional null is true. BH rejection-event frequency equals FDR only under this experiment's global latent null. No power or mixed-null FDR claim is supported.

First 30 pairs assess only validity and runtime. All 30 must be valid and projected 2500-pair wall time must be ≤1800 seconds. The same 30 count toward the prespecified fixed total. No seed replacement, repeated fit, alternative initialization or significance-based stop is permitted. Any invalid pair or incomplete fixed total disables confirmatory acceptance.

## Uncertainty and decisions

Primary: Δ = FWER(total) − FWER(rest), with paired D∈{−1,0,1}. Report b=total-only, c=rest-only, Δ=(b−c)/n and sample-SD/√n MCSE. R=2500 bounds the true MCSE by .02; it does **not** promise a .02 confidence half-width. The preregistered primary 95% Hoeffding half-width is .0543241. Its conservatism is explicit: point estimate, repeat-clustered MCSE and intervals must be read together. Direction is established by the primary rule only if this interval excludes zero and the entire fixed design is valid.

Supplementary exact paired difference interval: write Δ=π(2q−1), where π is discordance probability and q is total-only probability conditional on discordance. Separate 97.5% Clopper–Pearson intervals yield joint coverage at least 95% by the union bound, including random discordance count (conditional CP coverage for q). Take extrema of the product over the rectangle. Zero discordances do not imply zero-width certainty.

Exact two-sided McNemar is `binomtest(b,b+c,.5)` (p=1 with no discordances; undefined without any valid pairs). It tests equal marginal event probabilities, **not nominal FWER control**, and cannot replace the primary rule after observing results.

Each method has a 97.5% exact CP interval (Bonferroni joint ≥95% for two methods). Under a complete valid fixed design, upper≤.05 supports scoped control, lower>.05 supports scoped inflation, otherwise inconclusive. Failure envelopes allow every missing outcome to be either rejection or nonrejection. Empirical MCSE=0 is never treated as proof of zero uncertainty.

## Reproducibility

`results/metadata.json` records plan, source hashes, commit and dependency versions. `records.jsonl.gz` preserves every repeat's seed, data-content hash, raw LR/p diagnostics, decisions, validity and timing; its gzip round trip and SHA256 are verified. Data hash uses row-major uint8 response bytes followed by uint8 group bytes, with shape and ordered column names stored alongside. All first30 matrices and every failed matrix are compressed NPZ; remaining data are reproducible from immutable generator source/version/seed. No historical evidence is overwritten. See `REPORT.md` and `results/summary.json` for actual execution, not prospective validation claims.
