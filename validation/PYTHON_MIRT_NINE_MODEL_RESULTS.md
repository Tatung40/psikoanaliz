# Python mirt: nine-model result screens

## Root cause and scope

The dedicated `MTK: Python mirt …` route checked `FitResult.parameters`, which does not exist in Python mirt 1.2.0. Successful fits were therefore hidden behind an API warning. The verified public interfaces are `FitResult.coef()`, diagnostic attributes and `fit.model.parameters/probability/information`. Rasch also lacked a dedicated sidebar route. Polytomous routes preferred dichotomized `scored_df`, losing ordinal levels or nominal labels; they now prefer original data.

A shared, read-only `LibraryReport` exposes metadata, parameters, response curves, information, warnings and errors. Existing binary mappings and B14 probability/information helpers are reused. No calibration algorithm or scientific threshold changed. Downloads include convergence, estimation and usage status in summary/parameter tables. Download clicks do not rerun calibration. Data, items, category labels/count and estimation changes invalidate cached results before rendering any table, graph or download.

## Model matrix

| Screen | Actual calibration and coefficients | Prior screen failure | Current outputs |
|---|---|---|---|
| Rasch | Library 1PL/Rasch alias, a=1, b free, θ~N(0,1) | No route | Summary, a/b table, ICC/IIF/TIF/SEM, CSV |
| 1PL | Public 2PL equality constraint: one shared free a, item b | Wrong result API | Same binary outputs |
| 2PL | Native a/b | Wrong result API | Same binary outputs |
| 3PL | Native a/b/c; MAP default, ML selectable | Wrong result API | Same binary outputs and usage warnings |
| 4PL | Native a/b/c/upper d; MAP default, ML selectable | Wrong result API | Same binary outputs and usage warnings |
| GRM | Native a and ordered physical thresholds b | Wrong result API/data preference | Summary, parameter table, CRC/IIF/TIF/SEM, CSV |
| PCM | Native fixed a=1, unrestricted steps b | Wrong result API/data preference | Same polytomous outputs |
| GPCM | Native a and unrestricted steps b | Wrong result API/data preference | Same polytomous outputs |
| NRM | Native category slopes A and intercepts D; category0 A=D=0 | Wrong result API/data preference | Long category table, CRC/IIF/TIF/SEM, CSV |

Rasch here is distinct from the application's native free-latent-variance Rasch. The new shared-free-a 1PL is explicitly opt-in; adapter defaults and all existing callers retain the library's fixed-a 1PL. Earlier fixed-a validation evidence does not certify the shared-free-a specification. No raw intercept is silently relabelled as difficulty; the displayed physical b/steps are native Python library parameters. NRM labels preserve the declared reference order and are not achievement scores.

## Reproducible verification

`tests/test_python_mirt_nine_results.py` fits each family once per pytest session and reuses each genuine fit for table, curve, CSV, UI and failure tests. Existing deterministic fixtures are sliced to the first 300 respondents and six items, without new simulation:

- Rasch/1PL/2PL: `tests/fixtures/irt_b10/data_irt1pl_ok.csv`.
- 3PL/4PL: `tests/fixtures/irt_b11/data_irt3pl_ok.csv` / `data_irt4pl_ok.csv`; library default MAP priors.
- GRM/PCM/GPCM/NRM: `validation/evidence/p5_browser/poly_ordinal.csv`, four categories; NRM relabelled A/B/C/D without changing codes.

The tests check exact native parameter values/order, fixed/equality/reference constraints, parameter count for shared 1PL, response curves and information against model APIs, CSV round-trips, actual Streamlit tables/four graphs/four downloads, unchanged-input reuse, stale output removal, missing APIs and nonconvergence. Floating-point comparison tolerances here verify serialization/identical equations; they are not independently calibrated scientific acceptance thresholds.

## Scientific limits

These are reporting and integration checks, not independent R validation, parameter recovery studies or model certification. All nine small-fixture fits reported engine convergence; convergence alone does not prove identification or global optimality. Existing 3PL/4PL usage restrictions remain visible; the 300×6 smoke design is outside their studied parameter-recovery design. Numbers there are provisional/descriptive and must not be interpreted as validated item parameters. Default MAP priors are unchanged. Engine logLik/AIC/BIC at a MAP solution are labelled accordingly and do not imply the conventional ML optimum/model-selection equivalence.

No new SE/CI, DIF p-values, model-fit tests, person estimates or independent-reference claims are fabricated. Conditional SEM is `1/sqrt(test information)`, not parameter SE or posterior person SD. Missing diagnostic or coefficient APIs produce explicit errors; nonconverged estimates carry `Yakınsama=False` in exported parameter/summary tables. PR #63/#65 and their evidence remain unchanged.

## Executed small-fixture diagnostics

Python mirt 1.2.0; NumPy 2.4.6; pandas 3.1.0. Library defaults retained (EM, 21 GH points, max_iter=500, parameter-change tol=1e-4). These are UI smoke fits, not the GH61 matched-R protocol of PR #63/#65.

| Model | Estimation | Converged | Iterations | Free parameters | Engine logLik |
|---|---|---|---|---|---|
| Rasch | ML | True | 13 | 6 | -1077.204411533 |
| 1PL | ML | True | 13 | 7 | -1077.147614184 |
| 2PL | ML | True | 20 | 12 | -1073.788927260 |
| 3PL | MAP | True | 14 | 18 | -1167.235208292 |
| 4PL | MAP | True | 14 | 24 | -1204.333265262 |
| GRM | ML | True | 15 | 24 | -2302.787947947 |
| PCM | ML | True | 21 | 18 | -2344.343366315 |
| GPCM | ML | True | 45 | 24 | -2302.838597331 |
| NRM | ML | True | 35 | 36 | -2295.565542240 |

## Final checks

- Nine-model targeted checks: 27 passed; additional engine-failure/constraint guards: 11 passed. Sidebar/fingerprint checks also passed.
- One full regression: **925 passed, 2 skipped, 1 failed** in 940.21 s. The sole failure was the old source-string cache test requiring removed variable names (`old_categories`, `old_hash`), not a stale-result behavior failure. All nine actual calibration/report/UI checks passed in this full run.
- Replaced that brittle check with four no-fit Streamlit behavior cases (category count, nominal reference order, estimation, item selection); the updated file passed **5 tests**. No application code changed after the full run, and no second full regression was run.
- Python compilation and workflow YAML checks passed. The existing PR workflow now includes the new reporting and cache tests. Independent R validation, production deployment and merges were not performed.
