# NRM independent reference profile

The numerical budgets in `nrm_acceptance.json` are committed **before the first NRM R/Python calibration**. They specify reporting/computational accuracy, independently of the new fitted discrepancies. They are not externally calibrated clinical equivalence limits; a passing profile is MEASURED / NOT CERTIFIED, never population-wide scientific certification. Failed budgets remain failures and are not enlarged.

The existing PR72 driver, fixture, GH61 setup, artifact/reporting and PR71 person API are reused. Branches71/72/73 and main are not modified. R mirt1.48 and Python mirt1.2.0 remain independent engines. No new psychometric estimator is implemented.

Python NRM uses baseline-category A0=D0=0 and P_c=softmax(A_c*theta+D_c); free A∈[-5,5],D∈[-6,6], normal mean0/variance1,D1. R `nominal` is configured via its official values table: a1=1 fixed, ak0=0/d0=0 fixed, ak1..ak3 and d1..d3 free. Thus A=ak,D=d directly, including zero last-category slopes; no singular division by the default R score anchor is needed. Runtime checks must verify that R accepts/preserves this chart before any comparison is declared valid.

Categories retain explicit nominal labels; no thresholds, ordinal raw scores, Infit/Outfit/S-X² or Wright Map are inferred. Global theta reflection is a latent-orientation convention, not achievement direction. Each engine's reported orientation is chosen by its slope dot product with the same predeclared starts; near-zero orientation contrast blocks rather than forcing alignment. Raw coefficients are preserved. Curves/person theta are reflected consistently; likelihood and posterior SD are unchanged under the symmetric prior.

One fit per engine per attempt, same existing500×8 data/coding/starts/GH61/prior, ML/no item priors/no acceleration. Actual constraints, native/common logLik, IC identities, scores and convergence are retained. R person moments are evaluated independently from its fitted probtrace with identical GH61 normal weights, as in PR72; default fscores quadrature is not silently substituted. No ordinal score sum is used for persons.

Actual execution, per-category coefficients, probability/IIF/TIF/SEM/person comparisons, versions, hashes, tests and CI outcomes are recorded in the PR and its original artifacts. Same-parameter algebraic checks are distinct from separately fitted estimate agreement. If exact public-API identification/constraints cannot be established, classify BLOCKED and stop forcing the comparison.

NRM application and Python validation person scoring uses the supported `mirt.fscores(..., method="EAP", n_quadpts=61)` API. Its implementation uses the same normal GH quadrature class as calibration. Independent R posterior moments and a same-probability replay check test integration identities; they do not replace the Python library estimator. Posterior SD is distinguished from information SEM.

## Actual independent execution (2026-10-26)

[Run40265350318](https://github.com/Tatung40/psikoanaliz/actions/runs/40265350318) calibrated each engine exactly once at source commit `fa27d5e2a61614c6575f481dc8b4523cc9587a66` (tested merge `18faa62199822b946734dfd23d9cfde2c65ae8f3`). Python3.11.17/mirt1.2.0 and R4.6.1/mirt1.48 used the existing500×8 nominal fixture. R converged in80 iterations; Python converged in71. Both use48 free coordinates, fixed N(0,1), explicit GH61, ML/no priors, identical starts and bounds. Neither estimate is on an active bound; maximum constraint violation0. The R free-coordinate chart has identity Jacobian A=ak,D=d; a1=1 and baseline coefficients remain fixed.

| Maximum discrepancy | Observed |
|---|---:|
| Baseline-category coefficients | 6.41717e-6 |
| Common-grid respondent logLik | 2.91038e-10 |
| AIC / BIC | 5.58430e-10 / 6.22094e-10 |
| Category probabilities, reporting range [-4,4] | 2.17870e-6 |
| Category probabilities, full evaluation grid | 2.23115e-6 |
| Item / test information, reporting range | 2.42447e-6 / 3.95811e-6 |
| Information SEM, maximum relative difference | 9.02318e-7 |
| Library EAP theta / posterior SD | 2.77143e-6 / 3.58651e-7 |
| Same-R-probability posterior replay identity | 5.99520e-15 |

Common-grid logLik: Python−5016.860169148529; R−5016.860169148238. Native likelihoods agree with their common-grid values (largest residual1.82e-12); k/N and IC definitions were checked. Maximum absolute physical score: Python2.94975e-5; R3.87314e-7. These are observed scores, not a new stationarity acceptance threshold. Different stopping rules (Python likelihood increment, R parameter change) and optimizers remain explicit; convergence flags do not establish global optimality.

All predeclared engineering budgets passed without amendment (criteria commit `a6df444`, SHA256 `c2fe8f5c6ee61e183d87fceb7dff57eb42e02cd7538591b6dafc8b19e3161e13`). **Decision: MEASURED / NOT CERTIFIED.** The inherited generic comparison helper retains `SKIPPED` only for uncalibrated scientific tolerance decisions; comparisons actually executed and their numerical measurements are retained. The separate preregistered engineering checks all passed. No mandatory execution step was skipped in the independent-reference job.

The failed pre-calibration runs remain preserved. Runtime diagnostics confirmed that R data.frame assignment only reordered the named attributes: `names,class,row.names,itemtype` became `names,row.names,itemtype,class`. Each attribute value, column class, exact start and free flag is checked unchanged, then original attribute order is restored. A no-fit regression accepts ordering alone and rejects changed itemtype. No bounds, calibration algorithm, data or acceptance limit was changed.

Machine-readable evidence: [report.json](evidence/nrm_reference/report.json). [Original complete artifacts](https://github.com/Tatung40/psikoanaliz/actions/runs/40265350318/artifacts/12626292253) retain raw/canonical coefficients, category labels/codes, GH nodes/weights, curves, information/SEM/person comparisons, session versions, hashes, optimizer history and logs (ZIP SHA256 `777f09ea919c19b2654dadde50e2ee1c09552b07fdc8399413810eefd41c9c8a`). Earlier failures: runs40264604246 and40265046736. Raw evidence is never replaced with synthetic references.

Reproduction, with R mirt1.48/nloptr and Python requirements already available, uses a fresh output directory (no automatic retries):

```sh
PYTHONPATH=. python validation/poly_r_preflight.py --output nrm-preflight
PYTHONPATH=. python validation/grm_r_comparison.py --model NRM --preflight nrm-preflight --output nrm-preflight/nrm-fit --execute
```

Application safeguards: NRM person scoring uses the library API; nominal score sums are withheld. Focused AppTest checks unique IDs despite duplicate dataframe indices, CSV/Excel numeric identity, nonconvergence export suppression, category-edit invalidation, disabled Wright Map and no ordinal item-fit calls. PR73 output controls are preserved; GPCM test indexes account for its additional item-fit table without dropping export/identity assertions.

Limitations: one fixed fixture with four supported categories per item, no population-wide equivalence criterion, no item SE/CI or coverage certification, no proof of global optimum or validity for sparse/missing/other-category designs. Nominal-axis sign is conventional, not achievement order. Independent calibration uses explicit precise item-solver settings; this is not certification of every production default/setting. No unsupported Infit/Outfit/S-X² or Wright Map is produced. Broader scientific certification needs an independently calibrated validation protocol; numerical closeness and green CI alone are insufficient.
