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
