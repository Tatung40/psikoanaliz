# 5000 x 40: preliminary external parameter recovery audit

Source: local independent SciPy estimate CSVs from prior experimental run; 40 items. These are **not** PsikoAnaliz core estimates. The fitting convergence and missing-data handling of these CSVs have not yet been independently certified.

| Model | a MAE | b MAE | c MAE | d MAE |
|---|---:|---:|---:|---:|
| 2PL | 0.3305 | 0.3879 | N/A (fixed zero) | N/A (fixed one) |
| 3PL | 0.2009 | 0.2575 | 0.0666 | N/A (fixed one) |
| 4PL | 0.5164 | 0.3458 | 0.0718 | 0.0628 |

For 4PL the maximum fitted a is 11.597; the independent fit may be unstable. Correlation of estimated vs true a is only 0.198. This is a **WARN**, not evidence that the platform's 4PL engine is incorrect. External fitting constraints differ from platform bounds (platform a <= 4), so direct parameter comparisons require a common parameterization and constraints.

Next gates:
1. Run platform complete-data 1PL–4PL and missing-aware EM on the same harmonized binary matrix.
2. Match missing-data handling, quadrature, bounds, identification, convergence and seed across estimators.
3. Compare against external estimator and scalar independent marginal likelihood; check parameter recovery and repeated-start stability.
4. Do not release the experimental missing-aware EM as scientifically validated until these gates pass.
