# P3 closure: independently validated 3PL reference, explicit 4PL obstruction

## Decision

Only **3PL native S-X² with the constrained-Gaussian MAP reference** is SCOPED CERTIFIED.
The native statistic is unchanged; the original chi-square p-value remains suppressed.
4PL item-fit inference remains NOT CERTIFIED. No P1/P2 or 4PL score-set output is changed.

| Model / procedure | Null / power R | Pooled Type-I (cluster MCSE) | Worst simultaneous CP upper | Power (CP lower) | Refusal | Post-fit median / P95 | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| 3PL constrained-Gaussian S-X² reference | 6000 / 600 independent calibrations | .04980556 (.00082441) | .06894566 | .3366667 (.2989046) | 0 | .12356 / .18971 s | SCOPED CERTIFIED |
| 4PL constrained standardized residual Q, development only | 100 / 150 | .038333 (.005220) | .174677 (insufficient null precision) | .026667 (.007313); upper .066868 | 0 | combined candidates .27951 / .35432 s | NOT CERTIFIED; no confirmation |

The registered acceptance tolerance is simultaneous per-item Type-I upper ≤.07 at nominal alpha .05, not exact nominal size or FWER/FDR certification. The worst confirmed 3PL item rate is .059667. The standardized 3PL secondary also passed numerical gates but was **not prospectively selected and is not certified or exposed**.

## Reproducibility and independence

- Base main: 262f1d9ab95911f29b2014d3236f0f03ac22a090; PR195 is separate and untouched.
- Protocol committed first: 5be1faad9b40d27373b34ff7c887030bf64c89ca.
- Boundary/orthogonal fallback amendment: 4b89f7d152e1f95f08c86cee59be6e5324239d6d.
- Method selected before fresh confirmation: 88afec3341859b71ca970fb064b2d2159f382fe8.
- Canonical confirmation run: https://github.com/Tatung40/psikoanaliz/actions/runs/45808993982 .
  PR head 88afec3; actual checkout e978d102e86157c044f2bc50ae2fe6b28f6d0c1e.
- Null seed vectors [202611262001,3,r,0], r=0..5999; power [202611262002,3,r,0], r=0..599.
  These differ from P1 closure-K's **4PL** seed vectors; a common root integer alone does not mean a common stream.
- Python 3.11.18, Python mirt 1.2.0. Full package versions and native-bound source are attached to workflow artifacts.
- Each dataset calibrated once, with cold native MAP/GH61/default priors/max500/tol1e-4; all 6600 converged.
  No P1/P2 simulation or old failed campaign was repeated. New-method pilots reused all 500 original development fits.
- Native S-X² identity max 8.99e-13, below 1e-8.
- Independent Python-statistics/statsmodels arithmetic audit agrees with the separate NumPy/SciPy aggregator.
  Dataset, not person/item interval, is the Monte Carlo cluster. Refusals count as null rejections and power non-detections.
- Compact raw records: evidence/p3_empirical/CONFIRMATION.jsonl.gz (6600 seeds, hashes, native statistics, decisions, convergence and QP diagnostics).
  SHA256 fc60a71366c0323ad62a2569496806de876f753c5ee3552b13effc941ff4abd2.
  Full coefficients, pattern counts and logs remain in immutable canonical workflow artifacts.
- Read-only audit: python -m validation.p3_empirical.frozen_audit .
  Confirmation is deadline-guarded; archival CI never recalibrates or generates data.

## Method and independent checks

For grouped raw-score residual h and fitted MAP eta, the original S-X² equals N sum_g w_g (phat'h_g)^2.
A-emp differentiated the **full** residual functional using empirical score-group masses and the observed joint penalized Jacobian, not expected Fisher sensitivity. Scores, Hessian, priors and conditional-probability derivatives passed independent centered differences at two step sizes; a full-functional re-solve verified influence derivatives. Imhof tails matched an independent positive-weight gamma-mixture series, including noncentrality; deterministic null-space shifts were retained.

A-emp's differentiable approximation could not handle the 148/150 boundary power fits in 3PL; the standardized empirical version also showed null inflation in 4PL. These failures are archived, not promoted.

The successful fallback draws the joint marginal likelihood-score/residual Gaussian law over all 4096 patterns. The entire physical-coordinate MAP perturbation solves a box-constrained quadratic problem with A=I-Hprior/N, random likelihood scores and prior drift gradprior/sqrt(N). Raw residuals retain their deterministic means and all nuisance cross-covariances. An exact active-set solver was independently checked against scipy constrained optimizations and KKT ≤1e-7. The null approximation was **not assumed finite-sample-valid**; fixed-size independent confirmation evaluated size, power, grouping selection, boundaries and refusals together.

There are 1024 independent training draws and 4095 reference draws; p=(1+exceedances)/4096. Production uses an independent recorded seed, persistent through reruns. Neither the point estimator nor parameter uncertainty is changed.

Han, Z., Sinharay, S., Johnson, M. S., & Liu, X. (2023). *The standardized S-X² statistic for assessing item fit*. Applied Psychological Measurement, 47(1), 3–18. https://doi.org/10.1177/01466216221108077 .
Its Rao–Robson covariance standardization motivates the separately named Q candidate; published ML asymptotics were not treated as proof for penalized MAP.

## Why the registered 4PL power alternative is unsuitable

The historical implementation redraws theta independently for item 1; it does **not** double that item's slope on the common latent trait. It was retained exactly, without an easier replacement.

Let m=E[.1+.8 logistic(1.6(theta+1.25))], so .5<m<.9. A feasible native 4PL item has a=5,b=-6,c=.1,d=m, with all other items unchanged. Couple the same normal theta, other responses and item-1 uniform draw:
P(disagreement per person) ≤ .8 E exp(-5 theta-30) = .8 exp(-17.5).
For N=3000, total variation of the two full-data laws is at most 3000*.8 exp(-17.5). Any test controlling .05 **at this feasible null** therefore has power at the registered alternative at most **.050060264**, below the required >.10.

This is an analytic model-discrimination obstruction, not structural nonidentifiability or a global-optimization claim. The single reference null cell does not establish uniform level; passing that cell by flagging this other valid null would not make a scientifically valid general item-fit test.

Stable GH121/GH241 diagnostic KL is about 5.09e-15 for the explicit feasible 4PL null. The population MAP optimum is only a local numerical diagnostic (projected gradients recorded); no global optimum is claimed. The original GH241 3PL diagnostic encountered 0*log(0) in negligible tails; stable log-mixture evaluation corrected the quadrature diagnostic without reoptimizing or changing any data.

A new prospective **genuinely model-misspecified** 4PL power alternative is needed before useful 4PL certification. This campaign neither changes that alternative nor grants a low-power certificate.

## Production scope and safeguards

Native library 3PL MAP/N3000/J12/GH61/tol1e-4/max500/default cold fit only. The application cannot verify a user's true item design or latent normality; the new columns explicitly disclose that the certificate concerns the registered reference design. No J30, ML, alternate priors/optimizer or held-out item-fit claim is made.

Ordinary chi-square p remains NA. The calibrated p and decision have separate names; local-fit routes outside this exact native scope remain closed. Source/proof hashes, convergence, full covariance, KKT and statistic identity fail closed. Session caching binds actual response values, fitted coefficients, source/proof fingerprint and an independent persistent seed; withdrawal invalidates screen and CSV/XLSX/ZIP/DOCX together.

Protected byte-identical records include all **14 P1 certificates (including 4PL c/d score sets), six P2 certificates**, the native adapter, MAP estimates and old c/d Wald refusal policy. Scope-specific P3 certificate count increases 0→1 of two models; relevant P1/P2/P3 output groups 20→21. Other platform certificates are not recounted or changed.
