# GRM / PCM / GPCM / NRM — pre-registered Monte Carlo (Python mirt 1.2.0)

Script `validation/polytomous_mc_study.py` (design and criteria in the docstring, fixed before results);
evidence `evidence/p4_polytomous/`. Independent NumPy generators, k = 10, C = 4, theta ~ N(0, 1),
N ∈ {500, 2000}, 50 replications per cell (400 fits; 0 failures, 0 non-convergence). This replaces the
two-seed pilot of PR #62 for single-group calibration.

## Verified algebra

* Parameter mapping: `model.probability` reproduced from the reported parameters with the closed forms
  (GRM cumulative logistic, PCM/GPCM adjacent steps, NRM softmax with reference category 0): max
  difference 0 in all 400 fits; category probabilities sum to 1.
* Information: per-item `model.information(theta, item_idx)` equals the independent Fisher
  information sum_c (dP_c/dtheta)²/P_c (finite differences) to < 1e-9 relative; test information is
  their sum. (Without `item_idx` the library returns test information only.)

## Results (library default path → application adapter)

| Model | N | curve RMSE | parameter RMSE | 95 % CI coverage | EAP r(θ̂, θ) | criteria |
|---|---|---|---|---|---|---|
| GRM | 2000 | .0125 | a .070, b .065 | a .958, b .952 | .914 | met |
| PCM | 2000 | .0110 | steps .077 | .940 | .925 | met |
| GPCM | 2000 | .0125 | a .063, steps .072 | a .952, steps .966 | .938 | met |
| NRM (library default SE) | 2000 | .0153 | slopes .149, intercepts .140 | **.601 / .629** | .915 | **failed** |
| NRM (adapter, fixed) | 2000 | .0153 | slopes .149, intercepts .140 | .947 / .932 | .915 | met |
| NRM (adapter, fixed) | 500 | .0294 | slopes .283, intercepts .264 | .968 / .951 | .913 | met |

AIC and BIC chose the generating model in 50/50 GRM and 50/50 GPCM data sets (GRM vs GPCM, N 2000).

## Defect found and fixed

Python mirt 1.2.0 `fit_mirt(..., model="NRM")` with the default `se_method="auto"` reports its SEs as
"oakes" but they are ≈ 0.45 × the empirical SD (30-replication check: SE/SD .453, coverage .618).
Explicit `se_method="oakes"`, `"crossprod"` or `"sandwich"` give SE/SD .99–1.00 and coverage
.944–.948; for GRM/PCM/GPCM "auto" and explicit "oakes" are identical. The adapter now requests
"oakes" explicitly for NRM (`tests/test_nrm_se_fix.py` guards the fix and its scope).

## Completed in the application (B14 page)

EAP person estimates with SE (CSV, XLSX), per-item/test information and SEM (plot, CSV, XLSX),
GRM/PCM/GPCM information-criteria comparison with the nested PCM ⊂ GPCM LR test. Browser E2E: real
upload → fit → charts → parameter CSV, person CSV and XLSX downloads; reopened parameters and θ equal
a direct fit to < 1e-15 for all four models.

## Limits

Single group, normal latent distribution, C = 4, k = 10. Item-fit statistics of Python mirt
(S-X², infit/outfit) are shown as library output but were not validated here; NRM's latent axis sign
is the library's and is not a success direction.
