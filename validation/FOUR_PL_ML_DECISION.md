# Q3 — 4PL ML: fixture-scoped numerical decision

**Decision: NOT CERTIFIED; keep unsupported SE/CI disabled.** This is not a
permanent prohibition of every 4PL ML implementation and is not evidence of
structural nonidentifiability.

Data: unchanged `tests/fixtures/irt_b11/data_irt4pl_ok.csv`, N3000/J12,
SHA256 `ff503e39dfe0ac2c7683c10158248305280070b17133548fd6e43f6ab413ae4c`.
Model: standard-normal marginal ML, D1, no priors; original physical a/b/c/u
bounds. Original Python and actual R mirt1.48 solutions and R curves/logs are
losslessly archived in `evidence/r_34.json`. Original R nonconvergence remains
FAIL. Python's original convergence flag alone did not establish stationarity.

## Protocol and numerical repairs

- Before profiling: protocol commit `dbf19bcf6fc85138f0208e1832568960f02ece5c`.
  Ten fixed u values spanning [.5,1], including both original fitted u values;
  all47 nuisance coordinates optimized from the same complete saved R solution.
- All ten initial L-BFGS-B flags reflected relative objective stagnation;
  physical nuisance gradients failed1e-5. Preserved in `profile_uncentered.json`.
- Numerical amendment `ea9ab5e5abd1208b038f4fb771d06990a75c7b34`, committed before
  another execution, centered the scalar objective by a constant. Same statistical
  objective, derivatives, bounds, tolerances and grid. Ten failures remain in
  `profile_centered.json`; no optimizer flag was converted into a scientific PASS.
- Final strategy change, preregistered in `9ed26b10e44e7acebe0e2e2504a6859736184406`:
  bounded Newton score polishing of five preselected existing points only,
  using the existing observed-information implementation, positive-definite
  active face, condition<=1e8, nondecreasing logLik and decreasing projected
  score. No further repair attempts. Twenty L-BFGS profiles plus five polishing
  runs; ten unique grid values, at most25 point optimizations. Zero Monte Carlo.

Production algorithms and all inference/coverage/parameter boundaries are unchanged.
No independent R optimizer was newly executed; this is a deterministic profile of
the common likelihood already checked against actual R probability output.

## Results

| Fixed M12 u | M12 a | M12 b | M12 c | GH61 logLik | GH61 nuisance max score |
|---:|---:|---:|---:|---:|---:|
| .5 | 4.922105 | 1.192250 | .165772 | -23272.60371420531 | 1.679e-7 |
| .6 | 2.734011 | 1.460719 | .161311 | -23272.57555826200 | 5.283e-9 |
| .6198282615 | 2.579905 | 1.509034 | .160627 | -23272.57515051148 | 3.496e-8 |
| .6290267572 | 2.518257 | 1.530954 | .160325 | -23272.57527529594 | 1.570e-9 |
| 1 | 1.646970 | 2.210107 | .153353 | -23272.61282107274 | 1.788e-9 |

Five/five satisfy the unchanged raw physical nuisance score<=1e-5 at GH61;
the fixed u coordinate is excluded from nuisance stationarity. Active-face
minimum information eigenvalues .03037–.10711 and conditions1.52e5–5.22e5
are positive/within1e8. Boundary flags and full coefficient vectors are in
`q1_q3/profile_polished.json`; the u1 profile fixes u at its upper bound.

Across these widely separated u values, 2(max logLik−min logLik)=.0753411 while
a varies1.647–4.922 and b1.192–2.210: strong practical weak-identification evidence
for this fixture. A coarse nonuniform profile has nonzero curvature; it is not an
exact flat likelihood ridge. Local stationarity/curvature do not prove global maxima.
No chi-square cutoff, profile CI or Wald inference is applied to this diagnostic.

## Independent quadrature audit and precise failed conditions

At the same fixed fitted coefficients, GH101−GH61 logLik is .000206–.000883.
Despite these small objective differences, **zero/five** profiles meet the same
1e-5 physical nuisance stationarity gate on GH101. Max scores are .076515–.106394:
M9:c for u>=.6, M12:c at u=.5. Thus small logLik discrepancies alone cannot
certify quadrature adequacy. These are fixed-solution audits, not GH101 refits.

The existing physical-coordinate Hessian finding (minimum eigenvalue .037891,
weakest eigenvector99.14% M12; scaled condition2349.5) is consistent with weak
practical identification. It does not establish structural nonidentifiability;
scaling changes eigenvectors and the99.14% figure belongs to the physical mode.

Current failed conditions: GH101 stationarity, original R EM convergence and
existing estimator-specific CI coverage. Current passing conditions: the five
local GH61 nuisance stationarity checks, feasible original bounds, active-face
positive information and the independent archived R probability mapping.
No general recovery, global optimality, population inference or all-design claim.

Next deterministic step, outside this batch: a separately registered higher-grid
profile with an independent higher-density audit, preserving the same constraints.
Do not run another Monte Carlo or loosen1e-5 to resolve a quadrature failure.
