> Historical audit record. Final classification and later evidence: [README.md](README.md). Earlier pending/unknown statements retain their original scope.

# Single R 4PL refit: NOT CONVERGED

Real run **40150750060**, commit faf7d5a, artifact **12582730224**.
One R fit, initialized from the complete saved Python solution; no Python refit.
No further fits or iterations were performed for this diagnostic. Original
artifacts/reports remain unchanged. Reproduce with the existing diagnostic:
`python -m validation.r_mirt_phase1.m12_diagnostic --source EXTRACTED/refit --output cap.json`.

**Termination:** metadata records iterations=3000, converged=false, matching
NCYCLES=3000. R stdout and stderr are empty. The outer parameter-change
criterion did not establish convergence within its budget. The artifact has
no per-iteration parameter/likelihood history or per-M-step NLopt status;
therefore an exact inner optimizer failure/stagnation cause is not observable.
Do not infer a successful optimizer exit or fix this by raising the cap.

## Numerical agreement is separate from convergence

| Quantity | Saved Python | Final R refit |
|---|---:|---:|
| Shared 61-GH logLik | -23272.57537960827 | -23272.575150795634 |
| M12 a | 2.514326107 | 2.578947891 |
| M12 b | 1.531223054 | 1.509091508 |
| M12 c | .160284005 | .160617410 |
| M12 u | .629026757 | .619828261 |

R minus Python logLik is **+0.000228812638**, not .0029 in this artifact.
Maximum M12 ICC difference is .008999804; M12 information difference .014767052.
R surpasses the saved likelihood, but the M12 parameters still differ and this
cannot be labeled converged or PASS. Both solutions retain the same bounds;
M9 a=5 and several g=0/u=1 remain active in both. M12 itself is interior.

## Physical stationarity and curvature

Final R M12 marginal physical scores (a,b,c,u):
(.000594331, .000421230, .001694851, -.000565118).
The native logit(u) score is -.000133165: the previous M12 endpoint suppression
(-3.98e-9) is gone. M12 endpoint flattening is therefore not an adequate
explanation of the current cap exit. Other exact-bound logit coordinates remain
flat, so their physical feasible scores must be checked separately.

A box-tangent score removes only ascent components blocked at exact active
bounds. Maximum absolute feasible score is .001694851 (R), versus .004187508
(saved Python); neither is declared numerically stationary by an invented
cutoff. A fixed feasible all-item ascent probe yields a tiny positive logLik
change of 2.73e-10; decreasing only M12 u by 1e-6 yields +3.60e-10. These are
small likelihood evaluations, not refits, optimizer steps, or convergence tests.
They agree with the nonzero scores but do not justify an acceptance threshold.

The M12 conditional observed information, from central differences of analytic
scores with other items fixed, has physical-coordinate eigenvalues approximately
[.04569, 7.58999, 381.12177, 16285.91889]. This strongly uneven local curvature is
consistent with slow, coupled parameter drift while logLik changes little.
It is coordinate-dependent conditional curvature, not a full profile Hessian
or proof of a unique causal explanation. Python and R also use different outer
stopping definitions. Missing histories prevent attributing the cap exit solely
to outer EM versus inner SLSQP progress.

**Final classification: NOT CONVERGED.** Numerical agreement improved greatly;
stationarity, optimizer convergence and global optimality are unresolved. No
iterations, tolerances, algorithms, bounds, inference gates or production code
were changed. No speculative repeated fit is warranted from this evidence.

Source-audit qualification: see CONVERGENCE_TRACE.md. R 1.48 forcibly reports
converged=false at the cycle cap even if the final parameter-change check passes.
Without the missing final pair, its actual satisfaction is unresolved; neither
the gradient nor the between-solution likelihood difference determines it.
