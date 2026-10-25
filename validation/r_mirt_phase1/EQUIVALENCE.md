# Targeted 3PL/4PL audit: NOT COMPARABLE

Evidence: real R run 40146456589, commit 0364494, artifact 12581207421.
Original artifact is preserved on GitHub, not overwritten. No new fits or R
executions were performed in this audit. Both engines originally reported
convergence for both models; this does not establish equivalent optima.

Python mirt 1.2.0 `estimation/base.py::_ITEM_PARAMETER_BOUNDS` fixes physical
(a,b,c,u) to ([.1,5],[-6,6],[0,.5],[.5,1]); 3PL u=1.
The R artifact's `raw_parameters.csv` gives a1 and intercept d unbounded,
g/u bounded [0,1]. M9 in 4PL has Python a=5 (boundary), R a=17.4648953644
(outside Python's feasible region). Several Python 4PL asymptotes also hit
bounds. 3PL fits have no detected boundary within 1e-6 (diagnostic distance,
not an acceptance threshold).

Existing Python adapter: ML, normal mean0/variance1, EM21-point GH,
500 iterations, tol1e-4. R reference: same identification/D=1/ML,
EM61-point default equally spaced grid with normal density, 3000 cycles,
TOL1e-8. Python `quadrature.py` uses roots_hermite scaled by sqrt(2),
weights /sqrt(pi). Equal node counts alone would not align integration.
Python EM `_evaluate` sums respondent marginal log probabilities without
item priors for ML; R `03-estimation.R` sums frequency-weighted pattern
log probabilities. The common-grid calculation below uses respondent
likelihoods directly, independently of engine-reported likelihoods.

Exact bounds cannot be expressed by the existing R reference's independent
parameter boxes: b=-d/a requires -6a <= d <= 6a (positive a). Limiting a alone
would leave different feasible regions; fixed d bounds would also be wrong.
The Python public fit API exposes no bound override. No private monkeypatch,
post-fit clipping, MAP substitution or misleading partially aligned refit was
introduced. A validated coupled-constraint/physical-parameter R specification
is needed before an equivalent ML refit; hence NOT COMPARABLE.

`equivalence_audit.py` re-evaluates the preserved estimates, not new estimates:
61 identical GH nodes/weights for both marginal likelihoods; existing 81-point
[-4,4] grid for ICC/test information. Stable log probabilities avoid arbitrary
probability clipping. Parameter discrepancies and boundary diagnostics are in
`equivalence_report.json`, with source checksums. Original raw outputs remain
the authoritative evidence; the report is diagnostic, without PASS tolerances.

| Model | Common GH logLik Python minus R | Max absolute ICC difference | Max absolute test information difference |
|---|---:|---:|---:|
| 3PL | -0.0023880430 | 0.0093947459 | 0.0460328011 |
| 4PL | -0.1895211580 | 0.0963196456 | 6.1174434344 |

These differences do not isolate optimizer versus bounds versus quadrature
causes. SE/CI equivalence remains unvalidated. Production settings/gates unchanged.

Reproduce after extracting the original artifact into SOURCE:
`python -m validation.r_mirt_phase1.equivalence_audit --source SOURCE --output audit.json`

## Final feasibility check (base f4b9617)

Correction to the earlier scope: the **existing box-only reference** cannot
express the physical b bounds, but R mirt itself supports general inequalities.
The actual R evidence uses mirt 1.48. Its versioned source was inspected:

- https://github.com/cran/mirt/blob/1.48/R/EMstep.utils.R : `Mstep.optim`
  calls `nloptr::nloptr(..., eval_g_ineq=solnp_args$eval_g_ineq, ...)` and
  passes `optim_args` containing `longpars` and `est`. The public
  `nloptr_args` is routed internally to this `solnp_args` structure.
- https://github.com/cran/mirt/blob/1.48/R/utils.R : g/u values and bounds
  are converted with `logit`; exact 0 and 1 are replaced with -999 and 999,
  respectively; probability transformation is `plogis`.

`physical_b_constraint.R` supplies the feasible coupled inequality callback:
for positive a, b=-d/a in [-6,6] iff d-6a<=0 and -d-6a<=0. It reconstructs
fixed/free coordinates through `optim_args` rather than incorrectly treating
parnum as an index into the compressed free vector. No use of this candidate
callback or claim of R runtime verification is made in this commit.

The bound correspondence in native coordinates would be a in [.1,5],
d in [-6a,6a], g in [0,.5], and u in [.5,1] (3PL fixes u=1).
Python uses P=g+(u-g)*expit(a*(theta-b)), with physical optimization coordinates.
R's finite logit-coordinate domain does not mathematically include the same
closed g/u endpoints: plogis(-999)>0 and plogis(999)<1 over the reals. Floating
point rounding/underflow may produce endpoints, but must not silently be called
an exact smooth reparameterization; endpoint solutions are present in the
original 4PL evidence. This is an obstruction for the **built-in finite-logit
profile**, not a proof that no R mirt custom-item implementation can match it.

Therefore no verified full matched profile exists yet; **zero new fits** were
run. The historical common-grid numbers above remain historical and unchanged.
Before spending fit compute, validate either a physical-coordinate R custom
item (including boundary probabilities/derivatives), or explicitly justify a
numerical endpoint-equivalence contract. Then match GH nodes/weights and ML
settings in validation only. Same tolerance numbers alone do not prove identical
convergence criteria. The new callback must also pass an actual R 1.48 smoke
check before use. Status remains NOT COMPARABLE; no scientific PASS asserted.
