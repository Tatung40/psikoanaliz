> Historical audit record. Final classification and later evidence: [README.md](README.md). Earlier pending/unknown statements retain their original scope.

# 4PL M12: boundary stagnation evidence; no new fit

Source: real matched run **40149576359**, commit 2b3f675, artifact **12581672043**.
Original evidence remains unchanged. The new diagnostic only evaluates the
existing solutions on their original shared 61-GH grid; **zero refits**.
Reproduce: `python -m validation.r_mirt_phase1.m12_diagnostic --source EXTRACTED/matched --output diagnostic.json`.

Both engines reported convergence, not iteration-cap exhaustion: R928,
Python2741, cap3000. Common-grid logLik: R -23272.616806195085;
Python -23272.57537960827 (Python higher by **0.0414265868**).
The 3PL result remains unchanged (logLik difference 1.54e-7).

M12 satisfies a∈[.1,5], b∈[-6,6], c∈[0,.5], u∈[.5,1] in both engines.
R u=.9999999633 is near the upper boundary, Python u=.6290267572 is interior.
M9 a=5 is a separate active slope bound in both solutions. There is no new
bound mismatch. Original boundary flags are retained in the diagnostic JSON.

Analytic gradients of the full respondent marginal logLik were computed using
posterior-weighted category scores (all items jointly), in physical coordinates.
They were checked against independent central likelihood differences in a small
deterministic unit test; feasible one-sided likelihood changes corroborate them
on the actual M12 data.

| M12 score coordinate | Python solution | R solution |
|---|---:|---:|
| a | .002780104 | -.000007334 |
| b | -.001056563 | -.000008746 |
| c | .002151727 | .000103525 |
| u | -.000620287 | **-.108545740** |
| native logit(u) | -.000144745 | **-3.98099e-9** |

R's logit transformation multiplies the physical u score by u(1-u), severely
flattening the gradient near this boundary. At an upper-bound maximum a negative
physical u score is not sufficient for stationarity: moving inward is feasible.
Indeed, holding every other parameter fixed, decreasing R M12 u by .0001
**increases** logLik by 1.01554e-5; decreasing it by .001 increases it by
3.86201e-5. Thus this R point is **not a local maximum of the shared objective**.
The evidence supports premature convergence/boundary stagnation, not a claim
that two distinct stationary local maxima have been demonstrated. The exact
backend termination reason and history were not saved, so the causal role of
the internal stopping test cannot be proven from the flags alone.

Replacing only R M12 with the Python M12 parameters increases logLik by
.03497630. The straight interpolation is not monotone (a .1 step decreases it),
so a simple convex-ridge or global-optimum claim would also be unsupported.
Python's physical scores remain nonzero; its solution is better here, but is
not certified stationary or globally optimal.

Maximum M12 ICC difference on the existing [-4,4] grid is .32779403;
maximum test-information difference is .45663089. Large upper-tail response
curve differences despite a small likelihood difference support weak upper-
asymptote determination. No acceptance threshold or inferential gate changed.
This is **LIMITED / 4PL equivalence unresolved**, not PASS. A refit is unnecessary
to diagnose the nonstationary R point; none was spent. If numerical agreement is
required next, one R refit initialized from the saved Python physical solution
is justified, with the same bounds/grid and physical-gradient reporting.

## Prepared single R warm-start diagnostic (not yet executed)

The existing manual workflow now downloads immutable run 40149576359's artifact
and verifies its report/data/parameter/grid hashes against `m12_report.json`.
It schedules **one R 4PL fit and zero Python fits**, using the saved complete
12-item Python solution, not merely replacing M12. No retry, jitter, multistart,
changed tolerance, changed bounds or changed optimizer is introduced.

The explicit `matched-python-start` mode maps physical (a,b,c,u) to native
(a1=a, intercept=-a*b, g=c, u=u). The R mirt `pars` values remain physical for
g/u and are transformed internally by the package exactly as before, including
0/1. The existing R runtime preflight now checks imported initialization and
native values. The constant linear Jacobian's arithmetic check uses exact
integer probe coordinates; the actual optimization starts are untouched.

Command after downloading/extracting that artifact:
`python validation/r_mirt_phase1/harness.py --output NEW_OUTPUT --models 4PL
--matched --refit-4pl-from ORIGINAL/matched`.

Raw initialization/configuration, stdout/stderr, new R estimates/curves,
common-grid likelihood and signed difference from the unchanged Python fit,
M12 parameters/curve/information differences/physical scores, convergence and
boundary flags are artifacts. Python output is explicitly reused, with its
original versions recorded. This is a dependent optimization diagnostic,
**not a new independent validation experiment**. Agreement has no automatic
PASS cutoff; original evidence is never overwritten.

Actual execution/results remain PENDING until Actions runs. A missing/expired
original artifact or failed checksum stops visibly rather than recalibrating
Python or silently changing inputs.
