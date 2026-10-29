# 1PL and 2PL diagnostics — validation record (B10)

> **Arşiv notu (2026-10-29):** Yerel ikili MTK motoru (`core.irt`, `core.irt_mixed`, B10/B11/B12/B13 karşılaştırma altyapısı) kaldırıldı; ikili MTK kestirimi yalnız Python mirt ile yapılır. Bu belgedeki kaldırılan modüllere ait bölümler tarihsel kayıttır. Korunan modüller (`core.irt_mg`, `core.irt_mg34`, `core.irt_dif`, `core.irt_q3`) etkin DIF/yerel uyum ekranlarında kullanılmaya devam eder.

**Scope:** 1PL and 2PL only. New modules `core/irt_mg.py` (single/two-group MML),
`core/irt_q3.py`, `core/irt_dif.py`, `core/irt_globalfit.py`, `ui/irt_b10_view.py`, and three new
tabs on the MTK 1PL and 2PL pages. 3PL/4PL estimation, pages and diagnostics are unchanged
(new 3PL/4PL diagnostics: B11). B1–B9 modules are not rewritten.

## Audit (LOOP 1)

**Identification of the existing models (`core.irt.fit_irt`, unchanged).**
P(X=1|θ) = logistic(a_i(θ − b_i)), θ ~ N(0, 1) fixed, MML-EM on Gauss–Hermite nodes,
complete 0/1 data only (items with p ∉ (0.01, 0.99) rejected). 1PL: a_i = 1 for every item,
so it has k free parameters (b). 2PL: a_i ∈ [0.15, 4], b_i ∈ [−4, 4], 2k parameters.
EM stops at |ΔlogL| < 1e-4.

The B8 Rasch model (`core.rasch`) has a = 1 **and a free latent SD** (k + 1 parameters;
equivalent to a 2PL with one common slope). The 1PL is the Rasch model with σ = 1 imposed; the
two are not equivalent and are never treated as such: B10 reuses only model-agnostic helpers
from B8/B9 (Gauss–Hermite grid, score tolerance, BH-FDR, ETS-like classes, adjusted-Q3
centering, two plotting helpers).

| Analysis (1PL/2PL) | Before B10 | B10 |
|---|---|---|
| Calibration, ICC, IIF, TIF, EAP/MAP/ML abilities | existing | reused unchanged |
| Approximate Q3 (Model Tanıları tab) | existing: complete data, no adjustment, exploratory | kept; superseded by the new tab |
| Q3 + adjusted Q3, pairwise missing, ranking, heatmap, WLE/EAP, bootstrap | — | **new** (`core/irt_q3.py`) |
| Warm WLE for 2PL; EAP with missing responses | — | **new** |
| Two-group DIF (uniform 1PL; uniform/non-uniform 2PL), purification, LR + Wald, BH | — | **new** (`core/irt_dif.py`, `core/irt_mg.py`) |
| M2 / RMSEA2 / SRMSR with the model's own free parameters | — | **new** (`core/irt_globalfit.py`) |
| Item observed vs expected by ability strata, raw-score distribution, residual summary | only "Model p", "Artık RMS" | **new** (descriptive) |
| Rest-score Pearson, S-X² (mirt-compatible), OPG/Hessian SE, KKT | existing | unchanged, not re-validated in B10 |
| 3PL/4PL diagnostics; > 2 groups; polytomous; M2 with missing data; DIF on incomplete data in the UI (MTK page requires complete data); parametric-bootstrap global fit; anchor-selection-adjusted inference | — | **deferred** |

## Methods

### Estimator used by the new diagnostics (`core/irt_mg.py`)
Direct L-BFGS-B on the marginal log-likelihood with the analytic score; missing responses
skipped (MAR). One group: θ ~ N(0, 1). Two groups: reference θ ~ N(0, 1), focal μ_F and σ_F
free; per item `anchor` (a, b equal), `b` (b group-specific; a common or = 1), `ab` (both
group-specific, 2PL only). Converged = score norm ≤ 1e-6 per person and no parameter on a
bound; observed information = central-difference Jacobian of the analytic score; SEs refused
when it is not positive definite. `quadrature="gauss-hermite"` (production) or `"mirt"`
(61 equally spaced nodes on [−6, 6]; validation only).

### Local dependence (`core/irt_q3.py`)
Residuals x − P(θ̂) with each model's own IRF (1PL a = 1; 2PL item a) and the parameters of
the calibration shown on the page; θ̂ = Warm WLE (root of Σa(x − P) + J/(2I), J = Σa³PQ(1 − 2P))
or EAP under N(0, 1). Pairwise deletion, `min_pairs` (default 30), NaN for zero variance.
Adjusted Q3 = Q3 − mean off-diagonal Q3; reference ≈ −1/(k − 1). Parametric bootstrap
(≤ 500 in the module, ≤ 100 in the UI): θ ~ N(0, 1), the observed missing pattern, and
re-estimation of every replicate **with the production estimator and the exact settings of
the shown calibration** (`run_irt_calibration`); failures/non-convergence are counted and
excluded; Monte Carlo p = (1 + #)/(B + 1) for the maximum adjusted Q3.

### DIF (`core/irt_dif.py`)
For each testable item, with all other items in their current role: M0 (item constrained),
M1 (b group-specific), M2 (a and b group-specific, 2PL).

| | 1PL | 2PL |
|---|---|---|
| Flagging test (BH-FDR) | LR M1 vs M0, df 1 | omnibus LR M2 vs M0, df 2 |
| Decomposition | — (non-uniform DIF is not defined with a = 1: no item slope, curves cannot cross) | uniform LR M1 vs M0 (df 1), non-uniform LR M2 vs M1 (df 1) |
| Wald contrasts (joint observed information) | Δb = b_F − b_R, SE, z | Δa, Δb, SEs |
| Effect size (focal-weighted, probability scale) | signed E_F[P_F − P_R], unsigned E_F\|P_F − P_R\|; ETS-like A/B/C on Δb | signed and unsigned, crossing point θ* = (a_F b_F − a_R b_R)/(a_F − a_R) |
| DIF type (flagged items) | uniform | uniform if the non-uniform LR is not significant; otherwise crossing (θ* inside μ_F ± 1.96σ_F) or non-crossing |

Anchors: all-other-items start, iterative purification until the flagged set repeats; minimum
anchor count max(2, ⌈0.2k⌉). p-values are withheld when any of M0/M1/M2 did not converge or a
parameter is on a bound; SEs when the information is not positive definite; untestable items
(no or constant responses in a group) are only anchors; error for a group < 50 persons or a
disconnected design; warnings for groups < 200, size ratio > 4, few/unstable anchors. LR is
used for testing because it is invariant to the (a, b) vs slope-intercept parameterization.
A 2PL comparison is never shown on the 1PL page.

### Fit (`core/irt_globalfit.py`)
M2 from uni- and bivariate margins with Ξ from 4th-order moments and a Jacobian over exactly
the free parameters (1PL: b_1..b_k; 2PL: a_1..a_k, b_1..b_k; θ ~ N(0, 1) fixed), df = k + k(k − 1)/2
− q; withheld for missing data, df < 1, rank-deficient Jacobian, k > 30 or a non-converged
calibration. Classes: M2/p/RMSEA2 inferential (asymptotic); SRMSR, raw-score distribution,
strata observed–expected (with item fit plots), residual summary descriptive; existing S-X²
experimental (not re-validated in B10).

## Reference software and data

R 4.3.3, mirt 1.48 (github philchalmers/mirt a955235), installed from source in the validation
container. `scripts/mirt_irt_b10_reference.R` → `tests/fixtures/irt_b10/r_b10_results.csv`
(runtime ≈ 2 min); inputs from `scripts/irt_b10_inputs_for_r.py`. mirt conventions: 2PL
slope-intercept (b = −d/a); 1PL = mirt '2PL' with every a1 fixed at 1 (not itemtype 'Rasch',
which frees the latent variance); mirt default quadrature. TAM/eRm: not used (eRm has no 2PL;
TAM not installable here).

Fixtures (`scripts/make_irt_b10_fixtures.py`, fixed seeds):

| Case | Design |
|---|---|
| `irt1pl_ok` | 1PL, k = 10, N = 1000, θ ~ N(0, 1) |
| `irt2pl_ok` | 2PL, k = 10, N = 1200, a ∈ [0.6, 2.0] |
| `irt2pl_missing` | 2PL, k = 10, N = 1000, 10% MCAR |
| `irt2pl_ld` | 2PL, k = 10, N = 1000, shared N(0, 1) nuisance on M4 and M5 |
| `dif1pl_uniform` | 1PL, R 1000 N(0, 1), F 600 N(−0.4, 1.2²), k = 12, Δb M3 +0.6, M9 −0.5, 6% MCAR |
| `dif2pl_mixed` | 2PL, R 1200, F 800 N(−0.3, 1.1²), M3 Δb +0.6 (uniform), M7 a 1.164 → 0.264 (crossing) |
| `dif2pl_none` | same groups, no DIF |
| small/unbalanced, untestable, disconnected, unidentified linking, non-convergence | derived in the tests |

## Results

Tolerances are those asserted in `tests/test_irt_b10.py`; numbers are the observed differences.

| Check | Setting | Result | Status |
|---|---|---|---|
| IRF / parameterization | logit = aθ − ab; 1PL a fixed | exact | PASS |
| Score (gradient) | analytic vs central FD, 1PL/2PL × GH/mirt | rel ≤ 1e-6 (observed ~3e-9) | PASS |
| Single-group estimates | ours (mirt grid) vs mirt TOL 1e-10 | max \|Δa\|, \|Δb\|: 1PL 1.0e-8; 2PL 1.5e-7 – 4.0e-7 (incl. 10% missing) | PASS (1e-5) |
| logL at fixed parameters | ours vs mirt TOL = NaN | identical to 1e-12 (e.g. −6325.191799121) | PASS |
| SEs | observed information vs mirt Richardson vcov (delta method for b) | max rel diff 2.2e-9 (1PL) – 3.1e-7 (2PL) | PASS (1e-5) |
| Production EM vs exact MML | fit_irt GH-41, default tol 1e-4 | max \|Δb\| 1.8e-4 (1PL); \|Δa\| 7.6e-4, \|Δb\| 5.1e-4 (2PL); < 1e-4 with tol 1e-9 | documented |
| WLE | fixed parameters vs mirt `fscores(WLE)` | max diff 1.9e-6 – 2.5e-6 (mirt optimizer tolerance) | PASS (1e-5) |
| EAP | mirt grid vs mirt `fscores(EAP)` | max diff ≤ 7.1e-8 | PASS (1e-6) |
| Q3 | same θ̂, a, b vs mirt `residuals(type='Q3')` | max diff 4.4e-16 – 5.0e-16, incl. pairwise missing | PASS (1e-12) |
| Q3 planted dependence | irt2pl_ld | top adjusted pair M4–M5 (0.174) | PASS |
| M2, df, p, SRMSR | mirt parameters, mirt grid | 1PL 31.0289170 (df 45), 2PL 37.4926814 (df 35), LD 59.2862255 (df 35, p = .0063): identical | PASS (rel 1e-9) |
| M2 Jacobian | analytic vs FD, 1PL/2PL × GH/mirt; df = s − q | atol 1e-8 | PASS |
| RMSEA2 | convention | ours N, mirt N − 1 (mirt value reproduced) | documented difference |
| Round-0 DIF LR | ours vs mirt multipleGroup LR and mirt `DIF(scheme='drop')` | 1PL max 2.5e-5 (mirt EM tolerance; our base logL is higher); 2PL ≤ 8e-7; `DIF()` vs mirt manual LR ≤ 2.4e-6 | PASS (1e-4 / 1e-5) |
| Joint logL at fixed parameters | ours vs sum of mirt single-group TOL = NaN | identical (rel 1e-12) | PASS |
| Final purified model | parameters vs mirt multipleGroup | max diff 0.9e-3 – 1.2e-3, mainly σ_F (mirt EM stops early; our logL is higher) | PASS (2e-3) |
| Δb, Δa and SEs | vs mirt vcov (delta method) | 1PL M3 Δb .61657/.61661, SE .133527/.133524; 2PL M7 Δa −.857356/−.857357, SE .133378/.133378, Δb SE .248907/.248910 | PASS (1e-4) |
| Known uniform DIF (1PL) | GH production, BH .05, purification | flagged {M3, M9}; Δb +0.617, −0.385; stable after 1 round | PASS |
| Uniform + non-uniform DIF (2PL) | dif2pl_mixed | flagged {M3, M7}; M3 uniform (non-uniform p = .63); M7 crossing (Δa −0.857, θ* = 0.17); omnibus = uniform + non-uniform LR | PASS |
| No-DIF control | dif2pl_none | none flagged, 12 anchors; μ_F −0.286 (impact) | PASS |
| Small/unbalanced groups | 400 vs 60 | warnings "küçük", "dengesiz"; < 50 → error | PASS |
| Untestable item, disconnected design | constant in focal; no common items | item only anchor, no p; error | PASS |
| Unidentified linking | all items group-specific | information not PD → no SEs | PASS |
| Non-convergence | max_iter = 2 | no p-values, note shown, nothing flagged | PASS |
| Streamlit (AppTest) | 1PL page: calibration → Model Uyumu → Q3 → DIF; mis-selected group column never an item; settings-change notice; a new calibration hides old B10 results; bootstrap refits with the shown settings | — | PASS |

### Limited deterministic simulation (`scripts/irt_b10_simulation.py`)

| Study | Design | Result |
|---|---|---|
| dif1pl_null | 20 reps, 1PL, R 800 / F 500, F ~ N(−0.4, 1.2²), no DIF | per-item p < .05: 5.8%; BH flag 0.8%; datasets with any BH flag: 1/20 |
| dif2pl_null | 10 reps, 2PL, same groups, no DIF | per-item p < .05: 3.3%; BH flag 0%; 0/10 |
| dif2pl_power | 10 reps, M3 Δb +0.6 (uniform), M7 Δa −0.9 (crossing) | M3 detected 10/10 (typed uniform 9, crossing 1); M7 detected 10/10 (crossing 10/10); non-DIF items p < .05 6.0%, BH flags 0 |
| m2_null | 20 reps each, k = 10, N = 1000 | p < .05: 1PL 1/20 (median p .38); 2PL 1/20 (median p .71) |
| q3_null | 5 reps, 2PL, N = 800, B = 25 (production EM refits) | bootstrap p_max < .05 in 1/5 (median .39) |

Run time 6 min 51 s.

Not a Type I error or power study: too few replications for general claims.

## Performance (validation container, single process)

1PL DIF (k = 12, N = 1600) ≈ 2 s; 2PL DIF (k = 12, N = 2000) ≈ 14–27 s (50–71 model fits);
M2 and Q3 without bootstrap < 1 s; Q3 bootstrap ≈ 0.1–0.2 s per replicate with the
production EM (UI cap 100).

## Known limitations (NOT VALIDATED beyond the stated checks)

- DIF p-values are conditional on the data-selected anchor set; the selection step is not in
  the SEs or the χ² reference. Purification cannot recover when most items carry DIF in the
  same direction.
- The LR χ² reference is asymptotic; small groups and near-bound slopes weaken it.
- The DIF-type rule (non-uniform LR at α for flagged items; crossing inside μ_F ± 1.96σ_F) is
  a reporting convention, not a separate test.
- ETS-like classes (1PL only) translate the Mantel–Haenszel convention and are approximate;
  no ETS class is given for the 2PL.
- M2 needs complete data and k ≤ 30; its small-sample χ² calibration is only checked by the
  limited simulation.
- Q3 bootstrap: validated by construction and a 5-replication check only (labelled experimental).
- The production calibration's EM tolerance (1e-4) leaves parameter errors up to ~1e-3 relative
  to the exact MML optimum; the B10 diagnostics use the parameters shown on the page.
- S-X², rest-score fit and the OPG/Hessian SEs were not re-validated in B10.

## Reproduce

```
python scripts/make_irt_b10_fixtures.py
python scripts/irt_b10_inputs_for_r.py
Rscript scripts/mirt_irt_b10_reference.R          # ~2 min
python scripts/irt_b10_simulation.py              # several minutes
pytest tests/test_irt_b10.py tests/test_irt_b10_streamlit.py
```
