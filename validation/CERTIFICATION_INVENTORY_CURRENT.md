# Current scientific certification inventory (main d2977dd, 2026-11-26)

Categories: **NUMERICAL AGREEMENT** (matches an independent reference on fixtures), **DESCRIPTIVE IDENTITY** (algebraic/library identity), **SCOPED CERTIFIED** (preregistered Monte Carlo inference gates passed within a stated design), **NOT CERTIFIED / UNSUPPORTED** (inference shown closed or not shown), **UNIMPLEMENTED**. A scoped certificate is never evidence for user data outside its design.

| Family | Certified outputs | Scope | Evidence |
|---|---|---|---|
| Binary IRT point/IRF/information | 1PL/2PL/Rasch CERTIFIED; 3PL output-level; 4PL (tol ≤1e-10, MAP max_iter ≥50000) | binary_certification PROTOCOL A1–D | `evidence/binary_certification/` |
| Binary fit outputs | LogLik/AIC/BIC, Infit/Outfit identity, S-X² statistic (numerical) | fixtures | PR #183 |
| P1 parameter inference (14) | 3PL a/b/c and 4PL a/b one-step Wald 95% CI (5); one-step SE 3PL a/b/c, 4PL a/b/c/d (7); 4PL c/d C(α) score-inversion 95% sets (2) | N=3000, J=12, registered items, θ~N(0,1), MAP GH61 | `evidence/p1p3_analytic/P1_CERTIFICATION.json`, `evidence/p1_4pl_score*/` |
| P2 independent persons (6) | 3PL/4PL EAP recovery, equal-tailed posterior 95% (marginal, Bayesian), score Poisson-binomial inversion interval | N3000/J12; declared independent, disjoint IDs | `evidence/irt_closure/P2_CERTIFICATION.json` |
| P3 item-fit reference (1) | 3PL unchanged native S-X² with constrained-Gaussian MAP reference; original χ² p stays closed | N3000/J12/default cold MAP/GH61/registered truth/N(0,1); R6000 null+R600 power; worst CP upper .068946, power lower .298905 | `evidence/p3_empirical/CERTIFICATION.json`, `P3_CLOSURE_M.md` |
| P3 separate standardized Q (1) | 4PL covariance-standardized residual Q with constrained-Gaussian MAP reference; ordinary S-X² χ² p stays closed | N3000/J12/default cold MAP/GH61/reference truth/N(0,1); R6000 null, R600 each of3 alternatives; Type-I .033819, worst CP upper .044526; nonmonotone power .195, dependence .610; two-stage .0333 NOT CERTIFIED | `evidence/p3_4pl_closure/CERTIFICATION.json`, `P3_4PL_CLOSURE_N.md` |
| Rater reliability (30) | All 30 methods SCOPED CERTIFIED (simulation protocol) and NUMERICALLY VALIDATED (R references) | CERT_PROTOCOL designs, amendments R-A1, R-A2 | `rater_reliability/evidence.json` |
| RaschPy MFRM recovery (8) | global, items, thresholds, bivector, matrix, centrality, pseudo_halo, bistretch parameter recovery | examples/mfrm fixtures; not TAM/FACETS-equivalent | `raschpy_mfrm/evidence.json` |
| mfrmr MFRM (numerical) | facet measures, thresholds, model fit, latent, rater ranking CERTIFIED vs mfrmr MML; SE/CI/person/fit PARTIAL; interaction/ZSTD/4-facet NOT CERTIFIED | fixtures | `mfrm/evidence.json` |

| Binary global M2 references (2 new groups;24 relevant P1/P2/P3/global groups) | Unchanged native M2 with constrained-Gaussian MAP reference:3PL linear,4PL nonlinear. M2/RMSEA2/SRMSR/CFI/TLI numerical/descriptive outputs separately labelled; classical MAP χ² and RMSEA CI stay closed. | N3000/J12/reference truth/N(0,1), single cold MAP/GH61/max500/tol1e-4. R6000null+600/class/model; TypeI.045/.043333; upperCP.050554/.048795. Sensitivity:3PL dependence+secondfactor;4PL dependence only. No nonmonotone or4PL secondfactor certificate. | `evidence/binary_global_fit/CERTIFICATION.json`, `BINARY_GLOBAL_FIT_CLOSURE_O.md` |

## Explicitly NOT CERTIFIED (closed or labelled)
- 4PL c/d one-step Wald CIs (withdrawn #193: whole-box 46%/39%, informative coverage .9005/.9141).
- 3PL/4PL MAP sampling SE; MAP-centred Oakes/sandwich/bootstrap CIs (PR #189–#191).
- 3PL/4PL S-X² chi-square p; Method A corrected reference (3PL per-item upper .0720; 4PL power .06); B99 bootstrap INCONCLUSIVE (PR #192).
- Optimised C(α) kernel numerical equivalence (1e-8 T tolerance failed; inference re-certified by fresh confirmation 202611261301).

## Prioritised remaining scientific blockers
1. Ordinary4PL S-X² χ² inference and two-stage-shape sensitivity remain NOT CERTIFIED. The old independent-item alternative remains practically indistinguishable from a feasible4PL null (power bound≤.050061). PR #204 separately certifies standardized4PL Q only for the reference null design and predeclared nonmonotone/dependence detection classes; no general-design, FWER/FDR or weak-shape certificate. The PR #1963PL certificate remains unchanged.
2. 4PL item-1 c: informative sets infeasible below N≈52–62k (bound); other N/J/designs uncertified.
3. Scope transfer: every scoped certificate is tied to one registered design; no design-eligibility check is possible on user data.
4. C(α) score sets computed by explicit opt-in now persist in CSV/XLSX/ZIP without refits; old4PL c/d Wald intervals remain NOT CERTIFIED. Global M2 certificates do not extend to D4/multi-start/unknown profiles or unregistered designs.
5. mfrmr MFRM PARTIAL/NOT CERTIFIED outputs; polytomous R comparisons (open PRs #65–#74).
