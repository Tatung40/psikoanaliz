# 3PL and 4PL diagnostics — validation record (B11)

**Scope:** 3PL (Phase A, validated first) and 4PL (Phase B). New modules `core/irt_mg34.py`
(bounded single/two-group MML), `core/irt_dif34.py`, `ui/irt_b11_view.py`; backward-compatible
extensions of `core/irt_q3.py` (c/d in the IRF, Warm WLE for 3PL/4PL) and `core/irt_globalfit.py`
(M2 Jacobian with c/d, conditional M2, `global_fit_34`); three tabs on the MTK 3PL and 4PL pages.
1PL/2PL/Rasch pages and B1–B10 behaviour are unchanged.

## Audit (LOOP 1)

**Parameterization (core.irt.fit_irt, unchanged):** P(θ) = c + (d − c)·logistic[a(θ − b)],
θ ~ N(0, 1) fixed; 3PL fixes d = 1. mirt: P = g + (u − g)·logistic(a1θ + d_mirt) with a = a1,
b = −d_mirt/a1, c = g, d = u (mirt estimates g, u on the logit scale).
**Estimator:** MML-EM with an L-BFGS-B M-step inside a **box**: a ∈ [0.15, a_upper (default 4)],
b ∈ [−4, 4], c ∈ [0, 0.35], d ∈ [d_lower (default 0.65), 1]; no prior or penalty (constrained
ML, not MAP); optional multistart and EM + L-BFGS-B refinement (B6/B7); complete 0/1 data only;
boundary flags with tolerance 1e-3 (B1); Hessian SEs refused at bounds (B2).
**Consequence found in this stage:** parameters on the box are the rule, not the exception. Every
3PL fixture (N = 300–3000) and every 4PL fixture had at least one c (often 0 or 0.35), d (often 1)
or a (4) on a bound, also when the true c ∈ [0.15, 0.30]. Ordinary MML asymptotics therefore cannot
be assumed, and all inference below is conditioned on interior solutions.

| Analysis | 3PL | 4PL |
|---|---|---|
| Calibration, ICC/IIF/TIF, EAP/MAP/ML, S-X², Hessian/OPG SE, KKT | existing, unchanged | existing, unchanged |
| Q3 + adjusted Q3 with the full IRF, pairwise missing, ranking, heatmap | **new, validated** | **new, validated** |
| EAP with c/d | **new, validated (mirt)** | **new, validated (mirt)** |
| Warm WLE (J = ΣP′P″/(PQ)) | **new, validated (catR)**; differs from mirt's "WLE" by definition | **new, validated (catR)** |
| Parametric Q3 bootstrap (production estimator and settings) | new, experimental | new, experimental |
| M2 / RMSEA2 with c (and d) in the Jacobian | **new**; inferential only for fully interior fits; with bound-active parameters experimental, no p | **new**; same policy (in practice always experimental or unavailable) |
| SRMSR, raw-score, strata obs–exp with item plots, residuals | new, descriptive | new, descriptive |
| Two-group DIF: omnibus LR (df 3), single-parameter LR (Δa, Δb, Δc), Wald contrasts, purification, BH, IRF comparison | **new**; p-values only for interior items (asymptotic, limited validation) | descriptive contrasts only; **inferential DIF NOT VALIDATED** |
| Parametric-bootstrap LR for boundary DIF items; Δd inference; > 2 groups; missing data on the MTK page | deferred | deferred |
| χ² / Wald reference for parameters on a bound (c = 0, d = 1, a = 4) | scientifically unavailable (non-regular; χ̄² mixture) | scientifically unavailable |

## Methods

**Estimator for the diagnostics (`core/irt_mg34.py`).** Direct L-BFGS-B on the marginal
likelihood with the analytic score (verified by finite differences, rel ≤ 1e-6), the production
box, MAR missing data. Converged = projected-gradient (KKT) norm ≤ 1e-6 per person. A parameter
within 1e-3 of a bound (the B1 rule) is *bound-active*; `regular` = converged and none bound-active.
The observed information is computed over the interior parameters with bound-active ones held
fixed (conditional information; flagged), refused if not positive definite or condition number
> 1e10, and refused outright when two groups share no anchor item (a metric identified only by the
box is not identified). BLAS is limited to one thread inside these fits (10× faster on 2 cores).

**Q3.** Residuals x − P(θ̂) with P = c + (d − c)·logistic[a(θ − b)] (never the 2PL logistic);
θ̂ = EAP (default) or Warm WLE; pairwise deletion; adjusted Q3 as in B9/B10. Bootstrap replicates
are generated from the shown model (θ ~ N(0,1), observed missing pattern) and re-estimated with
`run_irt_calibration` and **the exact settings of the shown calibration** (box, multistart,
refinement).

**WLE.** Warm (1989): root of Σ(x − P)P′/(PQ) + J/(2I), J = ΣP′P″/(PQ), I = ΣP′²/(PQ); bisection
on [−12, 12], bracket-end estimates flagged. mirt's `fscores(method='WLE')` maximizes L·√I (root of
l′ + I′/(2I), the Jeffreys mode). The two coincide for 1PL/2PL only; for 3PL they differ by up to
1.26 logits on our fixtures. Warm's estimator is validated against catR 3.17 (`thetaEst(method="WL")`).

**M2.** Jacobian over the model's free parameters, columns b, a, c (, d), analytic derivatives of the
uni/bivariate moments verified against finite differences (atol 1e-8); rank and condition of Δ′Ξ⁻¹Δ
checked (≤ 1e10). If bound-active parameters exist their columns are removed (held fixed), q and df
are adjusted, and the result is labelled experimental without p or RMSEA2.

**DIF (`core/irt_dif34.py`).** Concurrent two-group bounded MML; reference N(0, 1), focal μ, σ free;
anchors share a, b, c (, d). Nested models for the studied item: M0 (equal), M_all (all its
parameters group-specific; omnibus LR, df 3 or 4), M_p (only p ∈ {a, b, c[, d]} group-specific; df 1;
hypotheses in the (a, b, c, d) parameterization — "a only" keeps b, not the slope-intercept d, common).
An LR p-value is given only when M0 and the alternative converged and **none of the studied item's
parameters is bound-active in either model** (other items' bound parameters held fixed, reported);
otherwise the LR is shown without p (descriptive). BH-FDR over available p-values. Items without p
whose LR exceeds the χ² critical value are removed from the anchor set by a documented heuristic
("Çapa taraması") — not a DIF test. Wald SEs from the conditional information for interior studied
parameters only. The response-function comparison (focal-weighted signed/unsigned P_F − P_R, maximum
|difference|, numerical crossings inside μ_F ± 1.96σ_F) describes the DIF independently of the test.
4PL: `inferential=False` (all-other-items anchors, no p, no purification).

## Reference software and data

R 4.3.3, mirt 1.48 (github philchalmers/mirt a955235), catR 3.17 (github cran/catR mirror), installed
from source in the validation container. Scripts: `scripts/mirt_irt_b11_reference.R 3PL|4PL`
(unbounded mirt; ≈ 13 min for 3PL), `scripts/r_irt_b11_boxed_wle.R 3PL|4PL` (mirt restricted to the
same box via lbound/ubound; catR WLE; ≈ 3 min for 3PL); inputs from `scripts/irt_b11_inputs_for_r.py`;
fixtures from `scripts/make_irt_b11_fixtures.py`.

| Case | Design |
|---|---|
| `irt3pl_ok` | 3PL, k 12, N 3000, b ∈ [−0.4, 2.2], c ∈ [0.15, 0.30] |
| `irt3pl_easy` | 3PL, k 12, N 2000, b ∈ [−1.2, 2.0] (c weakly identified for easy items) |
| `irt3pl_ld` / `irt3pl_ld2` | 3PL, shared nuisance SD 1.2 / 2.5 on M5–M6 |
| `irt3pl_missing` | 3PL, 8% MCAR |
| `irt3pl_c0`, `irt3pl_weak` | c = 0 for three items; N = 300, k = 8 (weak c) |
| `dif3pl` | R 1500 N(0,1), F 1000 N(−0.3, 1.1²); M3 Δb +0.6, M8 Δc +0.15, M10 Δa −0.9 |
| `dif3pl_none` | same groups, no DIF |
| `irt4pl_ok`, `irt4pl_ld` | 4PL, k 12, N 3000, d ∈ [0.88, 0.98]; LD on M5–M6 |
| `dif4pl`, `dif4pl_none` | R 2000, F 1500; M3 Δb +0.6, M8 Δd −0.2 / no DIF |
| small groups, disconnected, no anchors, non-convergence | derived in the tests |

## Results — Phase A (3PL)

| Check | Result | Status |
|---|---|---|
| IRF, parameter transformation, c/d constraints | exact; 0 ≤ c < d ≤ 1 enforced | PASS |
| Score: analytic vs FD (3PL, 4PL; GH and mirt grid; all constraint types) | rel ≤ 1e-6 (observed ~3e-9) | PASS |
| Observed information vs B2 Hessian (`core.irt_hessian`, B3-validated against mirt) | rtol 1e-4 | PASS |
| logL at mirt's estimates | identical (rel 1e-12) for all four single-group cases | PASS |
| Estimates vs unbounded mirt | not comparable: mirt leaves the box (c = 0.42–0.54 for one item in three cases) | documented |
| Estimates vs mirt restricted to the same box | max \|Δa\| 1.7e-3, \|Δb\| 2.2e-3, \|Δc\| 1.1e-3; our logL ≥ mirt's (≤ 1.3e-3 higher; logit-scale g cannot reach 0) | PASS (5e-3) |
| EAP vs mirt | ≤ 1.3e-7 | PASS (1e-6) |
| Warm WLE vs catR | ≤ 3.1e-5 for unique roots (catR root tolerance); 1 of 2000 patterns (irt3pl_ld) has several roots → NaN, counted | PASS (1e-4) |
| mirt "WLE" | equals our Jeffreys-mode reconstruction (≤ 1e-4); differs from Warm by > 1e-2 | documented difference |
| Q3 vs mirt `residuals(type='Q3')`, full IRF, incl. missing | ≤ 5.3e-16 | PASS |
| Q3 detection | strong dependence (SD 2.5): M5–M6 top (adj. Q3 0.23); moderate (SD 1.2): not top-ranked — the 3PL slopes absorb it | PASS / documented limit |
| M2 Jacobian and df | FD atol 1e-8; df = 78 − 36 = 42; conditional columns and df verified | PASS |
| M2 vs mirt `M2()` | **mirt 1.48 uses the 2PL derivatives (θPQ, PQ, 1 − P) for 3PL/4PL items** (`dP` method of class `dich`); reproducing that Jacobian gives mirt's value to 1e-9 (44.5358, 48.0627, 47.0215); with the correct Jacobian ours is 47.6328, 49.0652, 45.9253; SRMSR and df identical | ours PASS; mirt defect documented |
| Round-0 omnibus LR vs boxed mirt | max \|diff\| 3.0e-3 (24 item tests); base logL ours ≥ mirt | PASS (5e-3) |
| Same vs unbounded mirt | up to 9.2 apart where mirt leaves the box | documented |
| Joint logL at mirt's final estimates | identical (rel 1e-12) | PASS |
| Planted DIF (dif3pl) | M3, M8, M10: LR 35, 67, 47, all removed from the anchors; all three touch a bound in one group, so they get **no p-value** (screened, descriptive); signs of E_F[P_F − P_R] correct (M3 < 0, M8 > 0) | PASS (descriptive) |
| No-convergence, no anchors, small groups, disconnected design, 2PL rejected | no SE / no p / errors as specified | PASS |
| Streamlit (AppTest, 3PL page) | calibration → Model Uyumu → Q3 → DIF; group/ID never items; settings-change notice; new run hides old results; bootstrap refits with the shown settings | PASS |

**Limited simulation (`scripts/irt_b11_simulation.py`; not a Type I error study).** 3PL, R 1500 /
F 1000, no DIF, 8 replications × 12 items: 62 of 96 item tests had an interior studied item; of
these 3 (4.8%) gave p < .05 (mean LR 2.79 vs χ²₃ mean 3). 34 tests were withheld (boundary); 1 of
them (2.9%) exceeded the χ² critical value and would be screened out of the anchors. M2 under a
correctly specified 3PL (10 reps, N 2000, production EM + L-BFGS-B): 9/10 fits had bound-active
parameters (M2 experimental, 1/9 with p < .05); 1 interior fit (p = .76).

## Results — Phase B (4PL)

Phase B started after all Phase A targeted tests passed. Two-group mirt fits for 4PL do not reach
TOL = 1e-8 in practical time (a first run exceeded 30 min); Phase B two-group references use mirt
TOL = 1e-6 and ≤ 3000 EM cycles (recorded in the scripts), single-group fits TOL = 1e-10.

| Check | Result | Status |
|---|---|---|
| Boundary behaviour (bounded fit, N 3000) | 9 (irt4pl_ok) and 12 (irt4pl_ld) parameters bound-active, mostly d = 1, c = 0, a = 4; conditional information condition number 1.2e6 / 1.8e5 | documented (non-regular) |
| logL at mirt's estimates (single group; joint two-group final model) | identical (rel 1e-12) | PASS |
| Estimates vs unbounded mirt | not comparable: mirt runs to a = 4.3–12.3, c up to 0.45, d down to 0.58 (not converged) | documented |
| Estimates vs mirt restricted to the same box | our logL 3.3e-3 / 2.2e-3 higher; max \|Δa\| 0.012, \|Δb\| 0.042, \|Δc\| 0.011, \|Δd\| 0.005 (flat surface) | PASS (5e-2, logL ≥) |
| EAP vs mirt | ≤ 5.9e-8 | PASS |
| Q3 vs mirt (full 4PL IRF) | ≤ 3.9e-16 | PASS |
| Warm WLE vs catR | unique-root patterns ≤ 3.05e-5; the Warm equation has **several roots** for 54/3000 (irt4pl_ok) and 207/3000 (irt4pl_ld) patterns (3PL: 1/2000), where catR returns one root by its search path; we return NaN and report the count (detected by a 0.02-step scan) | PASS for unique roots; WLE experimental for 4PL, EAP default |
| M2 Jacobian and df | FD atol 1e-8; df = 78 − 48 = 30; mirt's 2PL-derivative Jacobian reproduces mirt's M2 (22.5148, 27.5709); correct Jacobian 21.5181, 35.8769; Δ′Ξ⁻¹Δ condition 4.4e10 / 6.4e8 | ours PASS; mirt defect as for 3PL |
| M2 policy | production fits always had bound-active parameters → experimental/unavailable, never inferential; null simulation 10/10 conditional (2/10 p < .05, not reported in the UI) | PASS (policy) |
| Round-0 omnibus LR vs boxed mirt (24 item tests) | max \|diff\| 5.9e-3, median 1e-3; base logL ours 3–4e-3 higher | PASS (1e-2) as descriptive statistics |
| Planted DIF (dif4pl, descriptive) | M3 (Δb +0.6): largest LR (50.5), E_F[P_F − P_R] = −0.107; M8 (Δd −0.2): **not recovered** — d̂ = 1 in both groups, LR 1.3; slope contrasts of null items up to ±2.5 | documented: upper-asymptote DIF weakly identified; Δ estimates unstable |
| 4PL inferential DIF | χ²/Wald references not defensible (boundary, weak identification); UI shows only descriptive contrasts, no p, no purification | **NOT VALIDATED** (by design) |
| Streamlit (4PL page) | helper test: DIF descriptive only ("Yalnızca betimsel", no p, no purification); one manual AppTest smoke run (not in the suite): tabs render, a non-converged calibration withholds M2 | PASS (helper) / smoke only |


## Performance (validation container, 2 cores)

3PL single-group bounded fit (N 2000–3000, k 12): 2–10 s. 3PL DIF (k 12, N 2500, GH 31): 13 fits per
round, ≈ 30–90 s in total (purification adds rounds); 4PL DIF descriptive ≈ 55 s (13 fits).
Multithreaded BLAS made these fits ≈ 10× slower on 2 cores, so `core.irt_mg34` limits BLAS to one
thread (threadpoolctl, optional). Q3 bootstrap: one production calibration per replicate (UI cap 50).

## Known limitations

- 3PL DIF p-values exist only for items whose parameters are interior; items that need them most
  (DIF often pushes c or a to a bound) are frequently descriptive only. A parametric-bootstrap LR
  for boundary items is deferred (cost: two bounded multigroup fits per replicate and item).
- The anchor screen for boundary items is heuristic; p-values are conditional on the data-selected
  anchors and on bound-active nuisance parameters held fixed.
- 3PL χ² calibration is supported only by the limited simulation above.
- M2 is rarely inferential for 3PL and practically never for 4PL (bound-active parameters, ill
  conditioning); the experimental conditional M2 has no established reference distribution.
- Q3 sensitivity is limited for 3PL/4PL (slopes absorb moderate dependence).
- mirt 1.48's `M2()` is not a valid reference for 3PL/4PL (2PL Jacobian); mirt's `fscores(WLE)` is
  the Jeffreys mode, not Warm's WLE, for 3PL/4PL.
- 4PL inferential DIF is NOT VALIDATED (see Phase B).

## Reproduce

```
python scripts/make_irt_b11_fixtures.py
python scripts/irt_b11_inputs_for_r.py 3PL && Rscript scripts/mirt_irt_b11_reference.R 3PL && Rscript scripts/r_irt_b11_boxed_wle.R 3PL
python scripts/irt_b11_inputs_for_r.py 4PL && Rscript scripts/mirt_irt_b11_reference.R 4PL && Rscript scripts/r_irt_b11_boxed_wle.R 4PL
python scripts/irt_b11_simulation.py dif3pl m2_3pl
pytest tests/test_irt_b11.py tests/test_irt_b11_streamlit.py
```
