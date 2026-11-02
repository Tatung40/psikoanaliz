# Lord/Wald — targeted independent reference protocol

User-predeclared chi-square relative discrepancy limit **1e-3**, unchanged.
Three existing B10 fixtures: dif1pl_uniform, dif2pl_mixed, dif2pl_none. Nonnull
anchors are the archived anchor lists; null anchors are M1–M8. The remaining
items form the test family. No new data, Monte Carlo or post-result anchor choice.
These fixture anchors are benchmarks, not a prospective anchor-selection study.

**Missingness identified before reference fits:** dif1pl_uniform has 1174 missing
responses in 859/1600 rows. The protected Python adapter rejects missing data.
For this paired reference exercise, both methods and both languages therefore
use the SAME explicitly complete-case cohort (741 rows); exclusions are recorded.
Other fixtures have 2000 complete rows each. No automatic imputation or production
adapter/missingness gate is changed. This is not evidence for general MAR/MNAR use.

Wald: existing M1/M2 MML, reference N(0,1), focal mean/SD free, fixed anchors;
61 equally spaced [-6,6] nodes with normal masses, existing observed information.
Lord: separate public Python mirt 1PL/2PL ML (fixed N(0,1)), GH61, tol1e-8,
max_iter3000, Oakes covariance; independent R mirt1.48 uses identical GH nodes
and weights. 1PL links only location (A=1); 2PL mean/sigma b-anchor link.
All anchors' sampling uncertainty and within-group cross-covariances enter
analytic full Jacobians. R uses independent numerical differentiation of the
same explicitly declared link; group samples are independent for Lord only.

Both engines must converge, remain off bounds, and supply positive-definite
free-coordinate and contrast covariances with condition <=1e8. R slope/intercept
covariances are transformed to physical a,b; direct raw a,d Wald is not the same
finite-sample statistic and is recorded separately when available. Parameters
and covariance discrepancies are reported as well as chi-square. No arbitrary
additional accuracy threshold is introduced for those secondary comparisons.

Fits execute once and are cached with data/settings. No silent retry, new seed,
threshold relaxation or failure deletion. Raw parameters, covariance, status,
versions and logs are retained. R unavailable means NOT EXECUTED; failed
comparison means FAIL, not certification. No general DIF-FWER certificate.

The initial preparation attempt reached native 1PL MML on the original missing
matrix, then the protected Lord adapter rejected missing responses. That unused
fit is not a reference result. The explicit common complete-case policy above
was fixed before any R results; the retained paired evidence contains exactly
three native and six separate-library valid Python fits.
