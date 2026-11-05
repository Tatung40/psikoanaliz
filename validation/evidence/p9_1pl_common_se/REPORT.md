# Q4 — free common-a 1PL SE and Wald CI coverage

**CERTIFIED only for the registered design/settings, conditional on required CI.**
This is not certification of every 1PL model, arbitrary a/b values, other latent
distributions or uploaded datasets. Fixed-a p6 evidence was neither repeated nor
pooled with this experiment. Q5–Q10 were not implemented.

## Preregistration and precision

Base main `41c0a56` contains merged PR142. Initial preregistration `b5d332f574068e03d9074183dcf3a1bb71efbc67`;
final executable protocol **`a1db06887affb3c3bbb4dbcc8aa420f3b82259e4`**, both committed before the first campaign draw.
The second revision hardened label/outside-box validation before execution; no
design, seed or acceptance threshold changed after observing campaign results.

Planned R300 would supply only300 independent shared-a intervals, not3600.
Its expected nominal coverage MCSE sqrt(.95*.05/300)=.012583 exceeds the unchanged
.004 requirement. Before execution, R was therefore fixed at4300: allowing5%
ineligible intervals, ceil(.93*.07/(.004²*.95))=4283 datasets are needed at the
worst Bernoulli variance inside the accepted coverage band. This is planning,
not an automatic guarantee; actual clustered MCSE was separately evaluated.

N1000, J12, true a1 common/free at estimation, b=linspace(-1.5,1.5,12),
theta iid N(0,1), D1, no item priors. Seeds are independent reproducible
`SeedSequence([20261029,4,index])`, index0..4299. Existing p6 `simulate()` is reused;
the new stream does not replay p6 fits or datasets. Fit: public Python mirt1.2.0
2PL equality API, all12 discriminations tied/free, latent mean0/variance1 fixed,
ML/GH21/tol1e-4/max_iter500. Library Oakes covariance; 95% Wald estimate±Z95*SE.

One campaign completed in175.29s with two workers and one numerical thread each.
No other Monte Carlo study or new R calibration ran. Versions and source/data
hashes are in `coverage.json` and the raw records.

## Correct covariance coordinates and independent reference

mirt returns expanded covariance **J V J'** with12 copies of one shared a.
Its expected nullspace was incorrectly rejected as unidentified information by
the generic helper. New `core.irt_common_a_inference` verifies the *declared*
all-item equality group, labels, parameter count and identical covariance rows/
columns; a selection S with SJ=I recovers V. This is no pseudoinverse, eigenvalue
clipping or new estimator. All a–b cross-covariances remain. Genuine singularity,
nonfinite covariance, condition>1e8, nonconvergence and boundary/outside-box
solutions remain rejected. The original inference helper and other models are unchanged.

Existing free-SD Rasch point identity is reused. A necessary deterministic
existing1000x10 fixture also independently checks the full covariance transform:
a=sigma; b=b_native/sigma. The Jacobian includes a's log-sigma derivative and
the b/log-sigma cross terms of the **full inverse** native observed information.
Max parameter difference .00047143; logLik difference .00001697; relative SE
difference .00022358 < existing .02 reference tolerance. Both fits converged;
independent covariance condition51.91. Full matrices are in `preflight.json`.
This is a native independent numerical check, not a newly executed R comparison.

## Results and uncertainty

4300/4300 converged, all4300 supplied valid covariance/CI; no fit failures,
constant items, withheld intervals, replacement seeds or omitted datasets.
All13 distinct parameter coordinates and pooled-b preregistered gates passed:
convergence>=.99, |mean bias|<=.05, coverage within.95±.02, eligible>=.95,
MCSE<=.004 and at least3600 eligible intervals. Failure-sensitivity bounds also passed.

| Coordinate/scope | Eligible | Coverage | Replicate-cluster MCSE | Bias / RMSE |
|---|---:|---:|---:|---|
| One shared a | 4300 | .950000 | .0033240 | -.0001271 / .0372568 |
| Each b_j | 4300 each | .947209–.956279 | max .0034105 | max absolute bias .0069665; all RMSE in JSON |
| Pooled b (secondary aggregate) | 51600 | .951240 | .0010307 | Individual-coordinate checks cannot be replaced by this pool |

Shared-a coverage exact95% CP[.943057,.956323]; Bonferroni simultaneous95%
CP[.939643,.959101]. Across all13 individual coordinates, the simultaneous
interval envelope is[.936608,.964778]. These were preregistered uncertainty
reports; they are not retroactively substituted acceptance thresholds.
Pooled-b cluster/t95% interval[.949220,.953261], using independent datasets,
not iid item intervals. Shared-a effective information sample size4299
(finite-cluster correction); pooled-b equivalent43660.99, with **4300 independent
datasets**, not43661 independent experiments. Average shared-a SE .0372840,
empirical SD .0372609, SE/SD1.00062. All per-coordinate bias, RMSE, SE, CP bounds,
MCSE and failure-sensitivity bounds are archived.

## Safety, UI and exports

The application uses the same independent-basis helper. Only the exact registered
observable settings, standard-normal latent specification, library version and
unchanged source identities authorize the scoped gate. True a/b/population
conditions remain assumptions: labels explicitly say the uploaded data are not
certified. Outside scope, common-a inference stays closed. The one shared a is
reported once in SE/CI tables and CSV/XLSX/ZIP; raw per-item coefficients remain.
Fixed-a1PL/Rasch, 2PL, 3PL/4PL and other unsupported gates are unchanged.

Tests cover the expected embedding nullspace versus genuine singularity, boundary
and missing-metadata rejection, cluster dependence, one common-a denominator,
interruption without refitting, source/raw integrity, a recorded-result AppTest,
and CSV/XLSX/ZIP equality. No campaign runs in CI. The explicit Q4 archived-only
marker preserves all evidence/browser checks and fails for unrelated/engine changes.
One final full regression is requested through the existing opt-in.

## Reproduce

```sh
# Frozen evidence, no fitting or simulation:
python -m validation.common_a_se_study --audit
python -m pytest -q tests/test_common_a_se.py
# Explicit campaign reproduction only; do NOT launch as routine CI:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
python -m validation.common_a_se_study --run --workers 2 \
  --state /tmp/q4-common-a-state-a1db068 \
  --protocol-commit a1db06887affb3c3bbb4dbcc8aa420f3b82259e4
```

The original durable checkpoint directory is retained in the execution workspace.
Completed seeds are reused; claimed-but-incomplete seeds become explicit
INTERRUPTED records and are never silently refitted. Raw JSONL is stored losslessly
as gzip/base64 parts with individual and reconstructed SHA256 identities. It
contains all4300 seeds, response-content hashes, estimates/SEs, fit diagnostics,
statuses and timings; endpoints are exactly reproducible from the fixed Z95.
The first three complete independent covariance matrices are additionally retained.
