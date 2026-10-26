# Python mirt multigroup DIF — MEASURED / NOT CERTIFIED

Python mirt **1.2.0**, Python **3.11.17**; actual library calibrations, not an independent reference. R is unavailable locally: **NOT EXECUTED**. No Type-I/FDR certification, global-optimum claim or inference release.

## Verified API and design

`fit_multigroup` and `multigroup_dif` signatures are preserved in `evidence/python_mirt_dif/known.json`. Factory and invariance implementation were inspected. Public `fit_multigroup` + `InvarianceSpec('strict', free_discrimination=..., free_intercepts=...)` retain H1 and each H0, their likelihoods, parameter counts, coefficients and latent distributions. Public `invariance_lrt` verifies eligible signed-LR pairs. No estimation algorithm is implemented here.

This reproduces the explicit-anchor **add** design: H1 frees the chosen families on all nonanchors; H0 equates the studied item. Untested parameter families stay equal. Reference N(0,1), focal mean/variance estimated under the anchors. Two explicit anchors are a conservative application prerequisite, **not** a proof of identification or sufficient sample size. Anchor equality and reference scale are verified from actual fit objects. No automatic purification/anchor selection.

| Model | Actual calibration / LR-statistic test | Restrictions |
|---|---|---|
| Rasch / 1PL | Yes, public `1PL`, a=1 | Shared-free-a 1PL unavailable in this API; differs from the single-group 1PL screen |
| 2PL | Yes | Location, slope or omnibus family |
| 3PL / 4PL | Yes | c/u remain equal; no c/u tests or inference; existing bootstrap gates unchanged |
| GRM | Yes | Common category coding, ordered model thresholds |
| PCM | Yes | a=1, location family only |
| GPCM | Yes | Step/slope families |
| NRM | Yes | Explicit nominal category mapping, category0 reference; no ordinal interpretation, DTF or Wright Map |
| RSM | No | Public multigroup model factory does not support it |

All models: **Wald, SE/CI-based inference, p/q and significance decisions disabled**. Multigroup results do not provide an observed-information Hessian/SE computation. Valid convergence alone does not establish regularity. Available invalid SE, negative/nonfinite LR, invalid df and nonconvergence are recorded; negative LR is never clipped. Requested BH/Holm/Bonferroni is recorded but **not applied** while p is suppressed. No claim of validated uniform/nonuniform decomposition for nominal categories or of FDR control after anchor selection.

## Measurements (existing fixtures; no R reference)

- Actual nine-route calibrations: Rasch/1PL, 2PL, 3PL, 4PL, GRM, PCM, GPCM, NRM. Small API checks may be nonconverged and are never interpreted as scientific PASS.
- Existing 2PL mixed-DIF fixture: N=2000, 12 items, GH41, tol=1e-5, max_iter=500, anchors all except M3/M7, omnibus. All three fits converged. M3 LR=48.96747793279064, df=2; M7 LR=52.00308238405705, df=2. Recovered directions: M3 focal b higher; M7 focal a lower. Parameters, likelihoods, constraints, versions and input hash preserved in `known.json`. This checks the known fixture, not general power.
- Small **12-seed** 2PL null diagnostic: N=600, J=6, GH41, location-only, five fixed anchors, one studied item. 12/12 valid converged fits; conditional chi-square rejection 1/12 (0.08333), exact binomial 95% CI [0.002108, 0.384796]. Too imprecise to certify size; does not evaluate BH FDR, other models or data-driven anchors. Production rejection rate is not estimated from its deliberately suppressed p values. Seeds, hashes and likelihoods retained in `null.json`.
- Public `multigroup_dif` add-design LR=0.00022455238195107086; retained public fit pair LR=0.0002759918706942699. Warm starts differ. Alternative-anchor LR=0.012375615978726273. Measurements only; no invented agreement tolerance (`api_parity.json`).

## Outputs and safeguards

Shared theta [-4,4], 81 nodes: library category probabilities, IIF and summed TIF; grid maximum absolute probability contrast (no severity cutoff). Ordinal/binary DTF is the difference of expected **coded** total scores, without latent-density weighting; not a test statistic. Real group/category counts, all H0/H1 coefficients, latent distributions, AIC/BIC and convergence diagnostics included. AIC/BIC are not absolute-fit validation. Curves from nonconverged fits are provisional.

CSV/XLSX/PNG/ZIP/Markdown carry the same restrictions. CSV/XLSX retain missing p/q. No fabricated Hessian, SE, item-fit statistic, DTF for NRM, or reference result. Original 1-based labels are preserved through PR #77 preprocessing; internal gaps and per-group missing categories are rejected. Data-content/schema/index and all analysis settings invalidate cached tables, curves and downloads. Missing responses are not imputed or silently removed; unselected group rows are explicitly counted.

## Reproduce / next scientific step

```bash
PYTHONPATH=. python -m pytest -q tests/test_python_mirt_dif.py tests/test_likert_category_origin.py
PYTHONPATH=. python validation/python_mirt_dif_check.py --output /tmp/dif-evidence
PSIKO_DIF_EVIDENCE=/tmp/dif-evidence PYTHONPATH=. python -m pytest -q tests/test_python_mirt_dif.py::test_known_dif_actual_library_and_report tests/test_python_mirt_dif.py::test_public_dif_parity_anchor_sensitivity_and_suppressed_invalid_exports
```

To enable inference later: independently align R multipleGroup/DIF specifications, validate regularity/boundaries, converge both fits, and establish design/model/test-specific size and multiplicity evidence. Current R validation of single-group fits or local DIF engines does not certify this new multigroup pipeline. Do not enable p/q based on model name or these small diagnostics.
