# Exact R mirt 1.48 convergence rule and prospective traces

Versioned source inspected (no new R execution or fit):
- https://github.com/cran/mirt/blob/1.48/R/utils.R : `hasConverged`
- https://github.com/cran/mirt/blob/1.48/R/EMstep.group.R : EM loop/cap
- https://github.com/cran/mirt/blob/1.48/R/03-estimation.R : stored histories
- https://github.com/cran/mirt/blob/1.48/R/extract.mirt.R : history accessors

The installed rule converts coordinates named `g` and `u` with `plogis`, then
requires **all(abs(previous-current) < TOL)**. Here TOL=1e-8, strictly '<',
not '<='. Other coordinates retain R's native scale, including the intercept
`d=-a*b`; this is not a change test on physical b. Fixed latent parameters
remain in the vector. It is neither a likelihood-increment nor a gradient test.

The EM loop calls this rule after the M-step, before saving that cycle into
EMhistory. Critically, **if cycles==NCYCLES it then sets converge=FALSE**, even
if the rule caused a break on that final cycle. Thus the existing 3000/false
metadata alone cannot prove that the final pair failed the numerical rule.
The old artifact has no previous-cycle parameters: final-pair satisfaction
is **UNRESOLVED**, while the package-reported status remains **NOT CONVERGED**.
This qualifies the causal interpretation in REFIT_CAP_DIAGNOSTIC.md.

Parameter stagnation, physical stationarity and objective progress are distinct:
- The old R point has nonzero physical M12 scores (maximum .001695), but those
  are not inputs to hasConverged. They cannot determine its Boolean result.
- R's final common-grid logLik exceeds saved Python by .000228812638. That is
  a between-solution difference, not the missing final EM likelihood increment.
- The old M12 u≈1 logit-gradient suppression concerned the internal optimization;
  the outer g/u parameter-change rule itself uses physical probabilities.
- Inner SLSQP exit reasons and per-cycle drift remain unknown in the old run.

Validation-only logging now enables `technical$storeEMhistory=TRUE` and NLopt
`print_level=1` (verbosity only). Algorithms, bounds, priors, grid, starts,
iteration caps and all numerical stopping settings are unchanged. NLopt's
per-M-step objective iteration output is retained in the existing R stdout log;
mirt does not expose all inner NLopt exit statuses, so none are fabricated.

`optimizer_trace.R` exports raw internal EM history, actual internal final
parameters, LLhistory (pre-M-step values), final evaluated logLik, OptimInfo,
and the installed hasConverged implementation. A last-pair CSV and summary
recompute the exact installed rule, keeping it separate from the package flag.
History length accounts for early-break versus completed-cycle ordering; an
unexpected alignment is explicitly unresolved. A future approved execution
can then distinguish parameter stagnation from objective progress. Existing
physical-score diagnostics remain separate; no acceptance cutoff is added.

A no-fit R regression in runtime_preflight.R checks strict inequality,
g/u physical conversion and both history layouts. **It has not run here:**
Rscript is unavailable. No workflow was dispatched and no new fit was run.
All historic artifacts/results remain unchanged.
