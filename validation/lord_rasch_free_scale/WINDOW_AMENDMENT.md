# Technical amendment 1 — register BEFORE expanded-window R fits

Original run41487568312/commit73fc86d completed four Python and four R fits.
All native fits converged; chi-square relative errors .000685514/.000763789
passed the unchanged .001 limit. However the R focal estimates have normalized
common-GH101 scores .000500814/.000556844. Original outputs remain immutable:
chi-square agreement alone is not common-likelihood stationarity.

Actual mirt1.48 source: R/utils.R makeopts assigns theta_lim=c(-6,6) if absent,
and R/03-estimation.R builds equally spaced seq(theta_lim, length.out=quadpts).
Source URLs: https://github.com/cran/mirt/blob/1.48/R/utils.R (1693,1748,1825),
https://github.com/cran/mirt/blob/1.48/R/03-estimation.R (663-664).
Hypothesis: finite-window Gaussian integration explains the residual score;
do not declare this proven until the new numerical check.

Verify these defaults/technical-option propagation against the ACTUAL installed
R namespace before ANY amended fit. Keep original data, starts, a=1/mean=0/free
SD, priors, item/variance bounds, Oakes, TOL1e-10, NCYCLES20000 unchanged.
Run exactly four R fits (groups R/F x points61/101), theta_lim=c(-12,12).
No Python refits: reuse all four original actual fits verbatim.
No optimizer/tolerance/acceptance relaxation. Save new R outputs under wide/;
retain every original file and the original workflow report.

In addition to the unchanged chi-square relative error<=1e-3 and valid covariance/
convergence/interior gates, require BOTH R and Python solutions to satisfy
max normalized score<=1e-5 on the independent common GH101 grid. This is the
existing numerical guard, now explicitly checked for the R reference too.
Report full covariance in (b,log SD), exact SD from CSV (diagnostic JSON rounds),
common-grid logLik and R61/101 density stability. If still invalid, retain
DESCRIPTIVE/NOT VERIFIED inference, report the obstruction; no repeated starts,
further window/iteration experiments or hidden failure conversion.
This amendment does not change the separately running GRM-FWER protocol.
