# RaschPy MFRM validation result (experimental)

Protocol: `PROTOCOL.json` (committed before comparisons). Engine RaschPy 1.4.0 (PyPI, Apache-2.0), 8 models.
Reproduce: `PYTHONPATH=. python -m validation.raschpy_mfrm.compare --write` (~20 s, deterministic).

| Model | Parameters | Persons/fit/probabilities | Rater sandwich SE | Cell bootstrap SE/CI | Fixture recovery max diff |
|---|---|---|---|---|---|
| global | CERTIFIED | CERTIFIED | NOT CERTIFIED (hidden) | UNSUPPORTED | 0.115 |
| items | CERTIFIED | CERTIFIED | NOT CERTIFIED (hidden) | UNSUPPORTED | 0.188 |
| thresholds | CERTIFIED | CERTIFIED | NOT CERTIFIED (hidden) | UNSUPPORTED | 0.283 |
| bivector | NOT CERTIFIED | NOT CERTIFIED | NOT CERTIFIED (hidden) | PARTIAL | 0.409 |
| matrix | CERTIFIED | CERTIFIED | NOT CERTIFIED (hidden) | UNSUPPORTED | 0.656 |
| centrality | NOT CERTIFIED | NOT CERTIFIED | PARTIAL (rater-side sandwich SE only) | PARTIAL | 0.368 |
| pseudo_halo | CERTIFIED | CERTIFIED | NOT CERTIFIED (hidden) | UNSUPPORTED | 0.230 |
| bistretch | NOT CERTIFIED | NOT CERTIFIED | NOT CERTIFIED (hidden) | PARTIAL | 0.455 |

* **V8 independent estimator** (`independent.py`; own pair counting, parameterisation and scipy optimiser; no RaschPy
  code): reproduces RaschPy's step grid for all 8 models on both recovery fixtures, unregularised and with RaschPy's
  default regularisation, max 7.3e-7 (tolerance 1e-4); default vs unregularised grids differ by ≥ 2.8e-3, so the check
  discriminates. The default-configuration comparison was named as the CERTIFIED requirement in the amendment; its
  tolerance (1e-4) is the preregistered V8 tolerance.
* **CERTIFIED** = V1–V3 (+V2b CSEM) identities + V4/V5 fixture recovery + V8 (both configurations, both fixtures);
  scope: these fixtures. bivector, centrality, bistretch remain NOT CERTIFIED because preregistered V5 fails.
* **V9 Monte Carlo** (30 reps; dense/linked; N 250/1000): all 8 models pass the bias and consistency gates. For the three
  failing models: bistretch:dense bias 0.016, SD ratio 0.53; bistretch:linked bias 0.036, SD ratio 0.53; bivector:dense bias 0.025, SD ratio 0.51; bivector:linked bias 0.039, SD ratio 0.50; centrality:dense bias 0.011, SD ratio 0.52; centrality:linked bias 0.027, SD ratio 0.55. Their V5 fixture misses are sampling variability of threshold-level rater cells (unbiased, SD ∝
  1/√N; linked design SD up to 0.16 at N=1000), not bias, identification or API error.
* **V10 bootstrap** (person-cluster, B=50, dense N=500, 30 reps): bistretch coverage 0.948, SE/SD 1.08; bivector coverage 0.954, SE/SD 1.05; centrality coverage 0.958, SE/SD 1.07 — gate passes; the app offers these cell SE/CI
  on demand for these three models only.
* **SE shown only with evidence**: rater-side sandwich SE / empirical SD must stay in [0.80, 1.25] for every cell in both
  designs (only centrality). Item, threshold and category SE have no coverage evidence and are hidden everywhere.
* Unchanged: identities exact (V1 ≤ 1.3e-15, V2 ≤ 5.5e-11, V3 ≤ 1.1e-15); SE/model selection/DIF/DRF/test summaries
  NOT CERTIFIED; Andersen LR and >1 rater facet unsupported; TAM differences descriptive only (≤ 0.026 logit).
* Remaining scientific limits: evidence covers the repository fixtures and the simulated conditions only (4 items, 4
  categories, ≤ 5 raters); V10 covers dense N=500 only; certification does not extend to other designs.

## Amendment 2026-11-22 (V11–V13; preregistered in PROTOCOL.json before running)

* **V13 separation / strata / reliability**: RaschPy's documented statistics recomputed from own WLE, residuals and
  information for all 8 models × 3 fixtures: max 1.5e-11 (tolerance 1e-6) → `test_stats` NOT CERTIFIED → **PARTIAL**.
  Caveat (descriptive): RaschPy's person RSEM is CSEM·√infit without a floor at 1 and is averaged over all persons while
  the variance excludes extremes; the Wright–Masters model reliability (CSEM, same person set) is ≈ 0.01 lower
  (e.g. 0.857 vs 0.868 on balanced_3facet).
* **V12 SE coverage** (100 null replications, dense/linked, N 250/1000, 400 fits per model, 0 fit failures): every
  model's sandwich SE groups pass (pooled Wald coverage 0.920–0.977, every parameter ≥ 0.85, SE/SD 0.80–1.20). Shown now:
  rater, item and threshold SE for all models with RaschPy rater SE; for stretch models only ω SE (λ SE not covered,
  blanked); bivector: item/threshold SE (RaschPy reports no per-cell rater SE; bootstrap SE/CI remains available).
  `se` → **PARTIAL** for all 8 models (simulation evidence under the null generating model only).
* **V11 fixture recovery with bootstrap SE (B = 200)**: bivector, centrality, bistretch still fail: rater P1's
  first-threshold cells in `unbalanced_linked` (and one bivector cell in `balanced_3facet`) lie 3.1–3.4 bootstrap SE from
  the generating value (|diff| 0.34–0.45). Combined with V9 (unbiased, consistent), this is a fixture-level sampling
  excursion of strongly correlated threshold-level cells, but the preregistered gate is not met → parameters remain
  **NOT CERTIFIED**.
* Held (no independent reference built): category widths (RaschPy bootstrap), model selection (AIC/BIC at non-ML RCL
  estimates), DIF/DRF → NOT CERTIFIED.
* Statuses of the 5 CERTIFIED models unchanged; R/TAM untouched.

## Amendment 2026-11-22b (V14–V16; preregistered before running)

* **V14 person CSEM** (all 8 models, dense/linked, N 250/1000, 0 fit failures): WLE ± 1.96 CSEM coverage 0.952–0.963
  in every cell → `person_csem` **PARTIAL** (null generating model only).
* **V14b stretch λ SE** (centrality, pseudo_halo, bistretch; 100 replications per cell): pooled coverage 0.947–0.976,
  SE/SD 0.99–1.14 → λ SE now shown.
* **V15 DIF/DRF: NOT COMPLETED** — the simulation process was terminated with the cloud session before writing results;
  no evidence → DIF/DRF remain **NOT CERTIFIED** (re-run: `PYTHONPATH=. python -m validation.raschpy_mfrm.simulation_v15`).
* **V16**: AIC/BIC declared **UNSUPPORTED** (RCL does not maximise the response likelihood) and removed from the app.
* Category widths: derived from thresholds (status follows parameters); RaschPy bootstrap width SE / 'Prop disordered'
  hidden. Bootstrap-method SE are never shown as validated (coverage evidence covers sandwich SE only).
* Reliability: RaschPy ('real' SEM) and Wright–Masters (model SEM) person reliability are now reported side by side.

## Amendment 2026-11-23 (V15 completed, V15c; preregistered before running)

* **V15** (global model, dense N=500, 60 null + 30 power replications, 0 errors): Wald z/p/BH identity exact (0.0);
  per-test null size DIF 0.042, DRF 0.072 (local) / 0.061 (CI; DRF bootstrap is not seeded inside RaschPy, so DRF rates
  vary slightly between runs — both pass); familywise flag rate 0.0; DIF power 0.97. → DIF and DRF Wald tables
  NOT CERTIFIED → **PARTIAL** (global model, dense N=500 scope).
* **DIF omnibus LR: NOT CERTIFIED** (null size 0.117; sensitivity runs 0.175 at N=250, 0.10 at N=750). Root cause: RaschPy's
  LR compares joint log-likelihoods at RCL (non-ML) estimates with plug-in pooled person locations, so Wilks' theorem
  does not apply and no chi-square reference exists. Independent-null rank calibration only holds for its own DGP;
  a dataset-specific parametric bootstrap needs ~100–200 refits (~16 s each) per analysis. Labelled "not calibrated" in
  the app. DRF omnibus passed its null gate (0.033) at this design only.
* **V15c Wald + BH global test** (reject if min BH-adjusted p < .05): null size DIF 0.000, DRF 0.067 (local) / 0.033
  (CI); DIF power 1.00 → **PARTIAL**, offered as the global DIF/DRF decision.
* Defect fixed: `simulation_v15.py` wrote a literal backslash-n after the JSON, making `simulation_v15.json` unreadable.

## Amendment 2026-11-23b — V17 holdout diagnosis of the bivector / centrality / bistretch recovery failures

200 independent holdout replications of each fixture design with the fixture generator itself (fresh seeds, 0 fit
failures; ~50 s on 2 cores). Matrix (CERTIFIED) as control.

| Model | max abs bias (bal / unbal) | bias gate | P(correct replicate fails V5) unbal | fixture tail prob unbal |
|---|---|---|---|---|
| bivector | 0.014 / 0.033 | pass | 0.145 | 0.53 |
| centrality | 0.009 / 0.027 | pass | 0.050 | 0.105 |
| bistretch | 0.009 / 0.035 | pass | 0.080 | 0.17 |
| matrix (control) | 0.031 / 0.048 | pass | 0.170 | 0.885 |

* **Root cause: no estimator defect.** V8 reproduces the estimator exactly; V17 shows every cell unbiased at the fixture
  designs; the V11 bootstrap SE match the holdout empirical SD (mean ratio 0.96–1.02). The failing cells are 2.3–3.2
  empirical-SD excursions of rater P1's first-threshold cells (where lenient P1 gives few 0 scores), and with 36–60
  correlated cells per model a correct estimator fails the per-cell `max(3 SE, 0.30)` rule on one fixture in 5–15 % of
  replicates (matrix 17 %; it passes only through its larger SE).
* **No code change** — nothing to fix in the estimator; regularisation and smoothing were ruled out (identical
  deviations with ridge = 0 and constants = 0).
* **Status unchanged: NOT CERTIFIED** — the preregistered fixture gate (V5/V11) is not met and acceptance criteria are
  not changed after the fact. Certifying these models would need a new criterion preregistered before evaluation on new
  fixtures (e.g. a family-wise rule); it is not adopted here.

## V18 — prospective family-wise recovery (all 8 models; preregistered before data generation)

Fixture generator, fresh seeds (20261124), balanced (300 × 3 raters) and linked (400, 2 of 5) designs, 200 replications
each (100 calibration / 100 confirmatory), 0 fit failures, ~90 s on 2 cores. Family-wise: Bonferroni critical z over all
K parameters with SDs from the calibration half.

| Model | max abs bias (bal / link) | SE ratio range | coverage (bal / link) | family-wise recovery (bal / link) | V18 | Status |
|---|---|---|---|---|---|---|
| global | 0.010 / 0.014 | 0.85–1.12 | 0.940 / 0.957 | 0.94 / 0.98 | pass | CERTIFIED |
| items | 0.011 / 0.033 | 0.87–1.19 | 0.942 / 0.952 | 0.95 / 0.94 | pass | CERTIFIED |
| thresholds | 0.010 / **0.055** | 0.87–1.13 | 0.949 / 0.947 | 0.90 / 0.96 | fail (bias, linked) | CERTIFIED → **PARTIAL** |
| bivector | 0.014 / **0.064** | 0.87–1.13 | 0.950 / 0.947 | 0.93 / 0.92 | fail (bias, linked) | NOT CERTIFIED |
| matrix | 0.033 / **0.082** | 0.84–1.19 | 0.943 / 0.946 | **0.88** / 0.93 | fail (bias, linked; FW, balanced) | CERTIFIED → **PARTIAL** |
| centrality | 0.010 / 0.033 | 0.84–1.14 | 0.945 / 0.951 | 0.95 / 0.96 | pass | NOT CERTIFIED → **CERTIFIED** |
| pseudo_halo | 0.011 / 0.030 | 0.85–1.16 | 0.940 / 0.957 | 0.95 / 0.97 | pass | CERTIFIED |
| bistretch | 0.013 / 0.039 | 0.84–1.19 | 0.940 / 0.949 | 0.94 / 0.96 | pass | NOT CERTIFIED → **CERTIFIED** |

* The historical fixture outcomes (V4/V5/V11) are kept in `evidence.json` as `historical_rule_status`; V18 does not
  reinterpret them. The bias failures are small-sample bias of threshold-level rater cells in the linked design
  (0.055–0.082 logit, above the 0.05 / 3 MCSE gate), consistent with sparse lowest categories for the lenient rater.
* Scope: null generating model (no true rater × item / threshold / stretch effects) and these two designs.
* CI: `.github/workflows/raschpy-v18.yml` re-runs V18 and requires the committed evidence to be reproduced.
