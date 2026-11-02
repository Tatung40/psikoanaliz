# Separate free-SD Rasch Lord reference — version 1

Registered before new fits; base main 00766b38b308e739e19ed77b154024997fb7df19.
Historical fixed-SD Lord outputs remain immutable and cannot authorize inference.

Use only the existing dif1pl_uniform binary B10 fixture, explicitly identical
741 complete cases from 1600 rows; the same recorded anchors/test family as PR130.
Separate groups: a=1, mean=0, SD free; A=1 follows from the common logistic unit,
not equality of latent SDs. No common-SD assumption or SD cutoff.

Python: existing core.rasch.fit_rasch, native defaults except GH61 and GH101,
max_iter=2000, score_tolerance=1e-6; no retries or alternative initializations.
Reconstruct full observed information using the existing Rasch _information,
invert jointly in (b,log SD), take its b block, never conditional inverse I_bb.
Native bounds b +-10 and log SD [log(.001),log(20)] remain unchanged.
Lord contrast and analytic full anchor Jacobian reuse core.dif_lord.
Require convergence, interior bounds, full and contrast PD covariance, kappa<=1e8,
at least two distinct anchors, disjoint target items.

R: actual mirt 1.48 itemtype='Rasch', mean fixed0, slopes fixed1, COV_11 free;
quadpts61/101, TOL1e-10, NCYCLES20000, ML without priors, SE Oakes.
Verify parameter table classes, constraints and actual free-parameter labels.
Use native quadrature for each engine, not silently equate their nodes/weights.
Physical b=-d, SD=sqrt(COV_11); transform the full inverse covariance before b
selection; independent numDeriv anchor-link Jacobian.

Preserve all fits/raw parameters/raw full covariance/versions/logs, even failure.
Compare b, SD, full and contrast covariance, chi-square/df/p, common-grid GH101
logLik and fixed-estimate GH61/101 scores. Pre-existing chi-square relative
limit <=1e-3 applies to both matched density pairs. No new parameter/covariance
acceptance limits. Cross-density differences are reported, not hidden.
Production quadrature guards require fixed GH61 vs101 |delta logLik|<=1e-3 and
GH101 max normalized score<=1e-5, inherited from the existing numerical gates.
No independent-reference PASS without actual R execution and valid fits.
Numerical agreement is not DIF-FWER, parameter recovery or CI certification.
