# 3PL/4PL uncertainty closure: bounded amendment E

Protocol commit: 63bdcbfc835c897d0e03b899b3708ce09d790f54. Base main: 34ca6ec. PRs #179–183 and all failed evidence remain unchanged.

## Confirmed blockers
- 3PL MAP posterior-curvature Wald intervals over-cover: b=.976, c=.975, MCSE≈.0025–.0026. Posterior curvature is not the sampling variance of a biased penalized estimator. The failed observed-information sandwich/crossprod alternatives are not repeated.
- 4PL MAP SE/CI calibration has not passed. Existing D4 point/IRF/information certification remains.
- Person posterior SD conditional on estimated items is not a frequentist person SE, does not include item-calibration uncertainty, and has no empirical coverage certification.
- Existing S-X²/df and mean-square identity evidence does not calibrate inferential p-values. The existing 3PL/4PL p-value gate stays closed.
- AIC/BIC evaluated at MAP do not establish classical ML model-selection evidence. Actual values remain numerically intact, with explicit interpretation fields in both production diagnostics/export paths.

## Demonstrated production scoring defect
The shared 161-point [-4,4] plotting grid truncates a non-negligible part of some posteriors. For 12 all-correct 3PL responses with a=1.4,b=2.5,c=.15,d=1 and N(0,1) prior:
- old-grid EAP=3.257053743602077, posterior SD=.4200868281935277;
- independent [-12,12] .002-grid integration: EAP=3.3563453600100868, SD=.5087901519260069;
- independent [-10,10] .002-grid integration agrees to 7e-15 in SD.
This is a deterministic mathematical counterexample, not a parameter recovery/coverage experiment or a fitted calibration.

Production 3PL/4PL scoring now calls the existing mirt.fscores EAP API at GH121/GH241 and requires both moments to agree within the preregistered absolute 1e-6 bound. It uses the same calibrated model, no refit, preserves person IDs and rejects nonconvergence/invalid scale/unstable quadrature. Both screen and cached CSV/XLSX/ZIP report paths use it; scoring-version changes invalidate old bundles. Other models retain their original path.

The targeted CI compares native scoring with independent real-line scipy.integrate.quad moments, preserving all-zero/all-one response patterns. Its artifacts contain actual numerical results. Local execution is NOT EXECUTED: the selected execution environment was still starting. No unexecuted test is claimed as passing.

## Why other quick fixes are not certified
A monotone re-expression of the same interval preserves its coverage event and cannot repair the observed overcoverage. A delta-method interval centered after transformation is a different method and requires independent coverage validation.
Wilks' regular ML profile likelihood result is not an automatic chi-square reference for penalized MAP points or boundary/asymptote ridges. Posterior profile curvature is not a frequentist pivot.
Replacing observed information by another unvalidated covariance or scaling SEs to the observed Monte Carlo rate does not establish holdout coverage. No thresholds, priors or solver tolerances were relaxed, and no repeated R timeout was launched.

References: Wilks (1938), regular likelihood-ratio theory; White (1982), curvature versus sandwich sampling covariance; Bayes' rule for conditional posterior moments. No new estimator or ad hoc interval calibration was substituted.

## Scientific decision
Existing item-estimate/IRF/information certifications are preserved byte-for-byte. No new parameter-interval, person-inference or item-fit-p-value certification is granted. Native conditional moment checks can verify numerical calculation only. Fresh coverage work needs a separately registered feasible study with dataset-cluster MCSE; the 30-minute limit does not justify certifying unknown coverage.

Software PR and CI artifacts provide the targeted verification record. ZERO remaining scientific blockers is **not established**. Merge must not be represented as completion of the unresolved inference certification.
