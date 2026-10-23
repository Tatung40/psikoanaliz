# GRM DIF: bounded, experimental evidence

Base `893d1cb` / PR #54. Completed 1PL/2PL modules, fixtures and simulations are
unchanged. The score-stratified screen remains a screen. This new method is
**EXPERIMENTAL**, not certification of every GRM design or all DIF families.

## Model and scope

`P(Y>=c | theta)=logistic(a*(theta-b_c))`; slopes positive and thresholds strictly
ordered by first cutpoint plus log positive gaps, never sorted after fitting.
Common ordered category codes 0..C-1, C=3..10; >=3 items, >=2 declared invariant
anchors and >=1 tested item. Every item/group/category needs >=5 observations by
default (configurable >=2); >=100 persons/group. These are operational safeguards,
not universal adequacy guarantees. Nonnumeric nominal labels, fractional scores,
infinities, degenerate/sparse categories, duplicate names and group-as-item fail.
Only selected groups and persons with observed responses enter the same likelihood
for all nested fits. NaN responses are skipped under MAR; exclusions are explicit.

Reference population N(0,1); focal mean and SD free. Both a and all b parameters
of fixed anchors link the scale. Other tested items remain group-specific nuisance
parameters. Declaring an anchor does not prove its invariance. No automatic or
data-selected anchor purification is implemented.

| Model | Studied item | LR hypothesis |
|---|---|---|
| H0 | a,b common | H2/H0: global, C df |
| Hb | b free, a common | H2/Hb: slope given free thresholds, 1 df |
| Ha | a free, b common | H2/Ha: thresholds given free slopes, C-1 df |
| H2 | a,b free | common alternative |

The b parameters are theta-scale cutpoints. In the slope-intercept convention
d=-a*b, sharing b while freeing a is a nonlinear constraint on d. **Ha is not
the usual R mirt linear d-equality hypothesis.** An eventual R reference must
respect this mapping rather than compare unlike tests. Each test family uses
separate BH across tested items; no joint FDR claim across the three families.
Unavailable p-values are not treated as nonsignificance.

## Numerical safeguards

Direct L-BFGS-B MML; stable log differences for category probabilities, analytic
scores, repeated-response-pattern sufficient counts, deterministic nested starts
and cached common alternatives. Bounds: a [.1,4], first b [-6,4], successive gaps
[.05,6], focal mean [-3,3], SD [.3,3]. A solution on a bound blocks inference.
Maximum absolute score/person <2e-6; observed information is the central-difference
analytic-score Jacobian and must be positive definite with min eigenvalue >1e-7
times max(largest eigenvalue,1). Iteration, gradient, bounds, eigenvalues and condition
number are reported. Optimizer success alone is insufficient.

Likelihood at the fitted parameters is audited with max(Q+20,81) nodes: absolute
logL difference <=1e-3, audit score/person <=1e-5. Failure withholds inference.
Negative/nonfinite nested gains are blocked using the previously verified LR guard.
Parameter/curve/signed-effect intervals use full joint observed-information covariance
and analytic delta gradients, including latent mean/SD uncertainty for focal-weighted
signed effects. Intervals are pointwise, not simultaneous.

## Independent numerical evidence

- Probabilities match the public Python mirt GRM model to 1e-12. Stable extremes,
  normalization and probability/physical-parameter derivatives are checked at 3/5
  categories; full marginal-score finite differences include missingness, shared
  thresholds and free slopes (4 categories).
- Marginal likelihood agrees with independent adaptive `scipy.integrate.quad`
  using direct cumulative-probability subtraction, including missing responses,
  to 2e-7 absolute logL.
- Eight actual **Python mirt 1.2.0 EM** fits: H0/Hb/Ha/H2 in null and planted cases,
  N600 reference / N400 focal, 6 items / 3 categories, latent focal N(-.4,1.2^2).
  Stored in `tests/fixtures/grm_dif/python_mirt_{null,both}.json`; no inference from
  that package's DIF wrapper is reused. All converged, parameter counts agree;
  logL differences <1e-5, physical coefficient differences <2e-4. Regression gates:
  logL <=1e-3, parameters/moments <=2e-3. Public API generator is
  `scripts/grm_dif_validation.py`; fixed results are reused by CI.
- Information curvature agrees with independent scalar objective second differences;
  category and signed-effect delta gradients match parameter finite differences.
  Reversing reference/focal preserves fitted likelihood and transforms slopes,
  cutpoints and focal moments according to the affine latent metric.
- R mirt was **not run**. Docker is available but had no cached R image; no large
  runtime/dependency download was made. The independent calibrator is Python EM,
  supported by analytic/adaptive-integral checks, not an invented R benchmark.

## Deterministic study (design-limited)

`scripts/grm_dif_study.py`: 100 null + 100 planted repetitions, deterministic
independent seeds, cached by source/version/config SHA-256. 6 items, 3 categories,
N600/400, 4 truly invariant fixed anchors; M5 all b +.6, M6 a -.8. Reference N(0,1),
focal N(-.4,1.2^2). All 200 repetitions were usable; 1,400 optimizer fits, ~154 seconds,
with the 4 pilot repetitions reused. No raw responses are exported.

| Quantity | Events / independent repetitions | Exact 95% Monte Carlo interval |
|---|---:|---:|
| Null: any global BH rejection | 7/100 | [.0286,.1389] |
| Null: any threshold BH rejection | 8/100 | [.0352,.1516] |
| Null: any slope BH rejection | 3/100 | [.0062,.0852] |
| Planted: M5 global detection | 100/100 | [.9638,1] |
| Planted: M6 global detection | 100/100 | [.9638,1] |
| Planted: M5 threshold detection | 99/100 | [.9455,.9997] |
| Planted: M6 slope detection | 88/100 | [.7998,.9364] |

Parameter interval coverage: 92–98%; signed-effect interval coverage: 92–95%.
Each rate has its own exact interval in `simulation_summary.json`; family audit
is separate. Focal mean/SD RMSE: .091/.115 (null), .097/.102 (DIF). The null intervals
are wide and allow material inflation. **Nominal FDR or general pointwise coverage
is not established.** Rates and coverage condition on valid fits; failed repetitions
are counted separately by the runner and never silently counted as no DIF.

## Reports and remaining gaps

Explicit ordinal/experimental acknowledgment is required in the new page; it has
no score-screen fallback. Stale data/settings hide old results. Item and group/ID
columns remain separate. Qualified Turkish DOCX/XLSX/BOM CSV and 300-dpi PNG in ZIP
contain methods, N/missingness, coefficients, parameter contrasts/intervals, nested
fit/statistics, convergence and quadrature diagnostics, overlaid category curves,
signed category/score contrasts and pointwise intervals. Formulas in labels are
escaped in spreadsheets; archive filenames are fixed; no respondent records.
Category-margin fit diagnostics are descriptive, not dimensionality/local-dependence
tests. References use verified ETS and official mirt citation metadata.

Remaining: R comparison with the correct nonlinear b mapping, heterogeneous category
calibration, data-selected anchors, broader sample/category/sparsity/LD simulations,
and stronger error-control evidence. Native implementation supports more categories
but independent EM and repeated-simulation calibration evidence here is 3-category.
Next user priority: PCM, then GPCM/RSM/NRM, then 3PL/4PL gaps. PCM/GPCM steps need
not be ordered; RSM needs truly shared steps; nominal categories remain unordered.
