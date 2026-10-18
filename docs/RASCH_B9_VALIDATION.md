# Rasch model fit, local dependence and measurement invariance — validation record (B9)

**Scope:** dichotomous Rasch model (B8 `core/rasch.py`, unchanged). New modules:
`core/rasch_dif.py` (two-group uniform DIF), `core/rasch_q3.py` (Yen's Q3),
`core/rasch_globalfit.py` (M2 and descriptive fit), Streamlit tabs on the "📏 Rasch Analizi"
page. B8 conventions are kept: latent mean 0, latent SD free, MML with Gauss–Hermite
quadrature, Warm WLE persons, MAR missing responses skipped in the likelihood.

## Method

### DIF / measurement invariance (`core/rasch_dif.py`)

| Element | Implementation |
|---|---|
| Model | Concurrent two-group marginal Rasch model (one likelihood, not two separate calibrations) |
| Identification | reference group μ_R = 0; μ_F, σ_R, σ_F free; anchor items share one b across groups |
| Scale alignment | explicit, through the common anchors inside the joint likelihood; group impact is μ_F, not DIF |
| Studied item | own b_iR and b_iF; all other items in the current anchor set are common |
| Estimand / sign | Δ_i = b_iF − b_iR (logit). Δ > 0: at the same θ the item is harder for the focal group |
| Anchors | start: all other items (leave-one-out); iterative purification (flagged items leave the anchor set until the flagged set repeats; max rounds configurable; minimum anchor count max(2, ⌈0.2k⌉)) |
| Test | Wald z = Δ/SE(Δ), SE from the inverse observed information of the joint model (refused if not converged or not positive definite) |
| Multiplicity | Benjamini–Hochberg FDR over the testable items |
| Effect size | ETS-like A/B/C on the logit scale (|Δ| 0.426 and 0.638, i.e. 1.0 and 1.5 on the Δ-MH scale with D = 2.35) — approximate guide |
| Quadrature | `gauss-hermite` (production, 41 nodes, θ = μ + σz) or `mirt` (61 equally spaced nodes on [−6, 6] with normal weights; used only to reproduce mirt exactly) |
| Guards | error: identical groups, α ∉ (0, 0.5), a group with < 30 persons, disconnected design (< 2 commonly answered items). Warning: group < 100, items untestable (no or constant responses in a group; anchor only), too few anchors, purification not stable |

This is **not** Andersen's conditional likelihood-ratio test (no CML is implemented); eRm's
CML Waldtest/LRtest are used only as agreement indicators. A significant Δ is a review
signal, not evidence of item bias.

### Local dependence (`core/rasch_q3.py`)

Residuals d_pi = x_pi − P(θ̂_p, b̂_i) with θ̂ = WLE (default) or EAP and the B8 item
estimates; Q3 = Pearson correlation of residuals over persons who answered both items
(pairwise deletion); NaN when fewer than `min_pairs` (default 30) common responses or zero
residual variance. Adjusted Q3 = Q3 − mean of all off-diagonal Q3 (Christensen, Makransky &
Horton, 2017); the reference value under local independence, ≈ −1/(k−1), is reported.
Optional parametric bootstrap (≤ 500, UI ≤ 200): simulate from (b̂, σ̂) with the observed
missing pattern, re-estimate the model and the persons in each replicate, record the maximum
adjusted Q3 → Monte Carlo p = (1 + #{max* ≥ max_obs}) / (B + 1), 95%/99% quantiles, and a
per-pair 95% quantile (descriptive). No universal cut-off is used as a test.

### Global fit (`core/rasch_globalfit.py`)

| Statistic | Class | Notes |
|---|---|---|
| M2 (Maydeu-Olivares & Joe), p | Inferential (asymptotically calibrated) | uni- and bivariate margins; Ξ from 4th-order moments; Jacobian in b_1..b_k **and σ**; df = k + k(k−1)/2 − (k+1); complete data only, k ≤ 30 |
| RMSEA2 | Inferential (based on M2) | √(max(M2 − df, 0)/(N·df)) with N (mirt uses N − 1) |
| SRMSR | Descriptive | observed vs model inter-item correlations |
| Raw-score distribution, observed vs expected | Descriptive | Lord–Wingersky; complete cases only |
| Item × WLE-stratum observed vs expected | Descriptive | standardized differences are not z-distributed (estimated θ, data-based strata) |
| Standardized residual summary | Descriptive | dichotomous residuals are not normal |
| Infit/Outfit (B8) | Descriptive | acceptable values are not evidence of overall fit |

No χ² reference is assumed for any descriptive statistic. No parametric-bootstrap global
fit test is offered in B9.

## Reference software and data

R 4.3.3, mirt 1.48 (github philchalmers/mirt a955235), eRm 1.0.4 (CML).
`scripts/mirt_rasch_b9_reference.R` → `tests/fixtures/rasch_b9/r_b9_results.csv`,
`r_b9_session.txt`; inputs from `scripts/rasch_b9_inputs_for_r.py`. TAM: **NOT TESTED** (not
installable in the validation environment). RaschPy (B8 second reference) has no MML
multigroup DIF or Q3 with comparable definitions and was not used in B9.

Fixtures (`scripts/make_rasch_b9_fixtures.py`, deterministic seeds):

| Case | Design |
|---|---|
| `dif_known` | k=12, b = linspace(−1.8, 1.8); R: N=900, θ~N(0,1); F: N=500, θ~N(−0.5, 1.2²); uniform DIF +0.6 (M3), −0.5 (M8); 8% MCAR |
| `dif_none` | same, no DIF (impact only) |
| `ld_pair` | k=10, N=1000, Rasch plus a shared N(0,1) nuisance on M2 and M3 (local dependence), complete |
| `rasch_main`, `rasch_missing` | B8 fixtures (misfitting slopes on M3/M8; 11.9% MCAR) |

## Results

| Check | Setting | Result | Status |
|---|---|---|---|
| Multigroup score | analytic vs central finite differences, both quadratures | max abs diff ≤ 1e-6·max|g| | PASS |
| Multigroup logL at fixed parameters | mirt parameters; ours (mirt grid) vs sum of mirt single-group `TOL=NaN` logLiks | −8613.898031573 vs −8613.898031573 (rel 2e-16); dif_none rel 4e-16 | PASS |
| mirt reported logLik | multipleGroup EM output | differs from its own fixed-parameter logL by 1.8e-4 (EM stops before the reported parameters); ours reaches −8613.897852 ≥ mirt | documented difference |
| Identification | mirt `est` flags | R: MEAN fixed, COV free; F: MEAN and COV free — same as ours | PASS |
| Item parameters | dif_known, purified anchors, mirt grid | max |b_R − b_R^mirt| 1.6e-5, max |b_F − b_F^mirt| 1.9e-5; μ_F −0.56008 vs −0.56005; σ_F 1.16219 vs 1.16108 (mirt EM tolerance) | PASS |
| Δ and SE(Δ) | vs mirt `vcov` (Richardson) | M3 Δ 0.412702/0.412714, SE 0.145256/0.145255; M8 Δ −0.478644/−0.478635, SE 0.140867/0.140866 | PASS (tol 1e-4) |
| Known DIF | production (GH-41), BH α = .05, purification | flagged {M3, M8}; Δ_M3 = +0.413 (q = .027), Δ_M8 = −0.479 (q = .008); stable after 1 round | PASS |
| No DIF with impact | dif_none | no item flagged (min q = .40); μ_F = −0.414 absorbs impact | PASS |
| Naive unaligned difference | separate calibrations, each with mean 0 | mean(b_F − b_R) > 0.35 (impact confounded) vs aligned mean |Δ| < 0.1 — the naive contrast is never reported | PASS |
| eRm CML Wald | agreement indicator | corr(z_ours, z_eRm) = 0.996 (both cases) | PASS (indicator) |
| eRm Andersen LR | agreement indicator | dif_known LR = 30.17, df 11, p = .0015; dif_none LR = 12.51, p = .33 | consistent |
| Q3 | ours vs mirt `residuals(type='Q3', Theta=WLE)` with identical b, σ, θ | max abs diff 4.7e-16 (ld_pair), 5.3e-16 (rasch_missing, pairwise missing) | PASS |
| Q3 planted pair | ld_pair | top adjusted Q3 = M2–M3 (0.117); mean Q3 −0.109 vs −1/9 | PASS |
| Q3 edge cases | zero variance, too few common responses | NaN (undefined), counted | PASS |
| M2, σ fixed | mirt grid, mirt parameters | ld_pair M2 53.13217474 (both), df 45, SRMSR identical; rasch_main 180.2263463 (both) | PASS (rel 1e-9) |
| M2, σ free | ours vs mirt | mirt omits σ from its Jacobian yet subtracts it from df (M2 identical to fixed-σ, df 44). Ours projects σ: 53.1268 (df 44). Ours verified independently against a finite-difference Jacobian implementation (both grids, rel 1e-6) | PASS (ours); mirt inconsistency documented |
| RMSEA2 | convention | ours uses N; mirt N − 1 (reproduced exactly with N − 1) | documented difference |
| M2 with missing data | rasch_missing | not computed; reason shown; classified "Hesaplanmadı" | PASS |
| Raw-score expected | Lord–Wingersky | expected sums to N | PASS |
| Streamlit | AppTest: B8 run → Model Uyumu → Q3 → DIF; group/ID columns never items; settings-change notice; new B8 run invalidates B9 results | — | PASS |

### Limited deterministic simulation (`scripts/rasch_b9_simulation.py`)

Not a Type I error or power study; too few replications for general claims.

| Study | Design | Result |
|---|---|---|
| dif_null | 25 reps, R 600 / F 300, μ_F −0.5, σ_F 1.2, no DIF | per-item unadjusted p < .05: 4.0%; per-item BH flag: 0.67%; any BH flag per dataset: 2/25 |
| dif_power | 25 reps, DIF +0.6 (M3), −0.5 (M8) | detection M3 15/25, M8 17/25; non-DIF items: p < .05 4.8%, any BH flag 1/25 |
| m2_null | 25 reps, k=10, N=800, σ=1.1 | p < .05 in 2/25; median p 0.33 |
| q3_null | 10 reps, k=10, N=600, B=50 | bootstrap p_max < .05 in 0/10; median 0.44 |

Run time ≈ 2 min 12 s in the validation container.

## Known limitations (not validated)

- DIF p-values are conditional on the purified anchor set; the anchor-selection step is
  not reflected in SE(Δ). Purification cannot recover when most items carry DIF in the same
  direction (DIF is then partly absorbed into μ_F).
- Only uniform DIF and two groups; no non-uniform DIF, no polytomous items, no multiple
  focal groups in one model.
- M2 needs complete data and k ≤ 30; small-sample calibration of the χ² reference is not
  validated beyond the limited simulation.
- Q3 bootstrap reference: validated only by construction and the 10-replication check
  (labelled experimental in the UI); the per-pair 95% quantiles are descriptive.
- The ETS-like A/B/C boundaries are a logit translation of the Mantel–Haenszel convention
  and are approximate.
- TAM was not tested.

## Reproduce

```
python scripts/make_rasch_b9_fixtures.py
python scripts/rasch_b9_inputs_for_r.py
Rscript scripts/mirt_rasch_b9_reference.R        # ~1.5 min
python scripts/rasch_b9_simulation.py            # ~2 min
pytest tests/test_rasch_b9.py tests/test_rasch_b9_streamlit.py
```
