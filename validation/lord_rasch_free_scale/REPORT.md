# Free-SD Rasch Lord — actual independent numerical evidence

Decision: **NUMERICAL REFERENCE PASS / NOT FWER CERTIFIED**.
Python3.11, existing native Rasch engine (NumPy2.4.6/SciPy1.17.1/pandas3.1.0);
R mirt1.48, actual Actions execution41489608564. Four original native fits reused
verbatim. Four default-window R diagnostic fits retained; four amended R fits
executed once, no alternative starts or repeated Python calibrations.

Same741/1600 complete-case cohort,12 items, original anchors, M3/M9 targets.
Both groups: fixed a=1/mean=0, independent free SDs. A=1 follows from slope
identification; no equal-SD assumption. Native b covariance is the b block of
the full inverse (b,log SD) information; R Oakes full covariance is transformed
to the same coordinates. Complete anchor uncertainty is propagated.

| Points | max relative chi-square | max absolute p difference | max R common-GH101 score | max common-grid logLik difference |
|---|---:|---:|---:|---:|
| 61 | 1.6709455e-05 | 1.2412956e-06 | 6.8901271e-06 | 1.4509396e-08 |
| 101 | 1.6697837e-05 | 1.2277028e-06 | 6.8901355e-06 | 1.4509851e-08 |

All fits converged and passed bounds/PD/condition<=1e8. Chi-square limit1e-3
and common-grid normalized score1e-5 were not relaxed. Default R [-6,6]
passed chi-square comparison alone but failed common-grid stationarity
(focal max5.57e-4). Actual R makeopts preflight verified the default and
the supported theta_lim option; [-12,12] reduced the residual below1e-5.
This supports finite integration-window truncation as the numerical cause.
Original/default-window results are preserved, including the diagnostic FAIL;
only the amended reference authorizes the new inference path.

Parameter, full nuisance covariance, contrast covariance, SD, density stability
and common-grid likelihood differences are in report.json. No invented
secondary acceptance limits. Full-precision raw CSVs, rather than rounded R
diagnostic scalars, supply SD/covariance comparisons. Effective settings
correct inherited unused fixed-SD labels; original settings are unchanged.

The new path reports CONDITIONAL INFERENCE / DIF-FWER NOT CERTIFIED. Legacy
fixed-SD 1PL results remain descriptive in UI/all exports. Source/data/evidence
identity changes fail closed. 2PL Lord and 1PL/2PL Wald are unchanged.
This is one fixture, not general Type-I/FWER, power, parameter recovery or
confidence-interval coverage certification; anchor invariance remains assumed.

Reproduction: see PROTOCOL.md and WINDOW_AMENDMENT.md. First-run calibration
commands are opt-in; committed evidence is verified without refitting by
`python validation/lord_rasch_free_scale/verify_reference.py` and
`python -m validation.lord_rasch_free_scale.compare --compare --wide`.
Original RDS/logs remain in the two Actions artifacts; CSV/JSON evidence,
transport digests and SHA256.json are committed. No model fits occur in
the cached-evidence verification path.
