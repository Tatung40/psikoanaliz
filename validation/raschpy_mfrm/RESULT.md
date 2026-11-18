# RaschPy MFRM validation result (experimental)

Protocol: `PROTOCOL.json` (committed before comparisons). Engine RaschPy 1.4.0 (PyPI, Apache-2.0), 8 models.
Reproduce: `PYTHONPATH=. python -m validation.raschpy_mfrm.compare --write` (~20 s, deterministic).

| Model | Parameters | Persons (WLE) | Fit | Recovery max diff | Recovery failures |
|---|---|---|---|---|---|
| global | PARTIAL | PARTIAL | PARTIAL | 0.115 | 0 |
| items | PARTIAL | PARTIAL | PARTIAL | 0.188 | 0 |
| thresholds | PARTIAL | PARTIAL | PARTIAL | 0.283 | 0 |
| bivector | NOT CERTIFIED | NOT CERTIFIED | NOT CERTIFIED | 0.409 | 4 |
| matrix | PARTIAL | PARTIAL | PARTIAL | 0.656 | 0 |
| centrality | NOT CERTIFIED | NOT CERTIFIED | NOT CERTIFIED | 0.368 | 1 |
| pseudo_halo | PARTIAL | PARTIAL | PARTIAL | 0.230 | 0 |
| bistretch | NOT CERTIFIED | NOT CERTIFIED | NOT CERTIFIED | 0.455 | 4 |

* CERTIFIED is unattainable: no independent implementation of RaschPy's regularised composite-likelihood estimator exists.
* V1–V3 identities (independent expected scores, stretch substitution, Warm WLE, infit/outfit) pass for all 8 models on
  all 3 fixtures (max 1.3e-15 / 5.5e-11 / 1.1e-15). First V3 run failed (3.5e-3) because the checker compared against
  3-dp rounded tables and dropped extreme persons; fixed in the checker (unrounded attributes, RaschPy's person set). The
  tolerance was not changed.
* bivector, centrality and bistretch fail the preregistered V5 null recovery (max(3 SE, 0.30)) on threshold-level rater
  cells (lenient rater P1, sparse low categories); they stay NOT CERTIFIED. RaschPy reports no per-cell SE for them.
* SE, test summaries, category widths, model selection (AIC/BIC at non-ML estimates), DIF and DRF: RaschPy-reported,
  NOT CERTIFIED (shown with that label).
* Unsupported (declared, not substituted): Andersen LR (disabled by RaschPy itself), >1 rater facet, TAM-style interaction LR.
* V6 (descriptive, different estimators): max |RaschPy global − TAM| for item/rater/threshold estimates: balanced_3facet 0.018, interaction_bias 0.026, unbalanced_linked 0.024.
* Edge cases fail closed via the reused `core.mfrm.prepare`: disconnected, duplicates, four-facet data, single rater,
  empty middle category; missing scores are excluded and reported. `sparse_category` fixture has no category < 10 obs.
* Performance (CPU container): fit 0.15–0.25 s per model after a ~7 s first import; bootstrap 200 ≈ 12 s; all plots +
  reports ≈ 15 s on demand; peak RSS ≈ 300 MB. Fits are cached per session (max 3), never shared across sessions.
