# Rating Scale Model — complete single-group chain

Module `core/irt_rsm.py`, page **MTK: RSM** (`ui/irt_rsm_view.py`). Separate from the fixed-anchor
RSM DIF engine in `core/polytomous_dif.py` (sigma = 1, two groups); never labeled PCM.

## Formulation and identification

P(X_i = c | theta) ∝ exp(sum_{s<=c}(theta − delta_i − tau_s)), common steps tau_1..tau_{C−1},
sum(tau) = 0, slope 1. Default identification: theta ~ N(0, sigma²), mean 0, sigma estimated
(Rasch family). Optional: sigma = 1 (comparable with libraries that fix the latent variance).
The partial credit model with item-specific steps is fitted only as the nested alternative of the
RSM-vs-PCM likelihood-ratio test (df = k(C−1) − (k + C − 2)).

## Implementation

Marginal ML on Gauss–Hermite nodes, analytic score (finite-difference check < 1e-7), L-BFGS-B;
convergence = score per person ≤ 1e-6 and no parameter on a box bound. SEs: inverse observed
information (central differences of the analytic score), withheld unless converged and positive
definite; derived last step by the delta method. Persons: Warm WLE (finite at extreme scores) with
1/sqrt(I) and EAP with posterior SD. Information I_i(theta) = Var(X_i | theta) (checked against
dE/dtheta). Fit: infit/outfit MNSQ and ZSTD with WLE, extreme persons excluded (with true theta the
MNSQ values are .93–1.07; with estimated theta short tests give values somewhat below 1).
Missing responses are skipped (MAR). Exports: XLSX (meta, locations, steps, latent, fit, persons,
RSM–PCM LR) and CSV; no respondent identifiers.

## Evidence (`evidence/p3_rsm/rsm_recovery_study.json`, design pre-registered in the script)

* Independent library check: under sigma = 1 the native fit equals Python mirt
  `EMEstimator(RatingScaleModel)` (a lower-level API not exposed by `fit_mirt`): 20/20 datasets,
  max |ΔlogLik| 5.4e-8, max parameter difference 1.6e-5 after the documented mapping
  (mirt fixes its first threshold at 0). Category probabilities agree to 1e-15.
* Recovery (100 replications per cell, 0 failures, all converged; delta, tau, sigma):

| N | k | RMSE δ | RMSE τ | RMSE σ | coverage δ / τ / σ | SE/SD δ / τ / σ |
|---|---|---|---|---|---|---|
| 300 | 10 | .103 | .071 | .071 | .948 / .950 / .960 | .99 / .99 / .95 |
| 300 | 20 | .108 | .045 | .058 | .915 / .958 / .960 | .94 / 1.08 / 1.01 |
| 1000 | 10 | .052 | .036 | .036 | .956 / .950 / .950 | 1.07 / 1.04 / 1.04 |
| 1000 | 20 | .057 | .025 | .034 | .932 / .958 / .930 | .94 / 1.04 / .89 |
| 3000 | 10 | .032 | .021 | .018 | .948 / .962 / .970 | 1.01 / 1.02 / 1.14 |
| 3000 | 20 | .031 | .015 | .020 | .950 / .945 / .930 | 1.00 / 1.04 / .94 |

  All N ≥ 1000 cells meet the pre-registered coverage [.925, .975] and |bias| ≤ .05 criteria;
  N = 300, k = 20 item-location coverage (.915) is just below.
* RSM-vs-PCM LR: Type I .06 (N 300) and .07 (N 1000), CIs include .05; power 1.00 when four item
  step patterns deviate by ±.6.
* Browser: real Chromium upload → fit → 4 charts → XLSX/CSV download; reopened δ and τ equal the
  direct fit to 4e-16 (`validation/browser_e2e_polytomous_dif.py`).
