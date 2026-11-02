# Fixed-solution GH81 diagnostic — no refit, no inference-gate change

**Answer:** the GH81 score exceedance does **not** disappear through GH161. The dominant coordinate is focal **log(SD)**. The evidence supports finite-order GH61 integration error shifting the stationary point, rather than an analytic-gradient, likelihood-scaling or floating-instability defect. No optimizer, Monte Carlo or model recalibration was run; prior p suppression remains unchanged.

## Reproducible scope

Base PR123 commit `2839aae`; seed2026110981, impact−.5; exactly the archived `failed_0881_minus.npz` and saved physical estimates. Source/data hashes and all146 protected prior files verified. Seven unique solutions cover eight item/hypothesis slots because H2 is shared. Free coordinates were reconstructed using native Design roles, positive slopes, ordered b0/log-gap thresholds, shared anchors and focal mean/logSD; the original encoded bits were not stored. Physical roundtrip error≤4.44e−16, GH61 saved logLik reproduced exactly, saved max scores reproduced within2.7e−18 (GH61) and5.2e−17 (GH81). No parameters were optimized or silently adjusted.

Native score is the gradient of **−logLik/N**, N=1000, in constrained encoded coordinates. Shared-anchor derivatives sum both groups before dividing by1000; focal derivatives also use totalN, not500. Total logLik score is −1000 times this vector. Reference theta~N(0,1), focal mean/SD and all identification constraints remain as saved.

## Max normalized score by fixed quadrature rule

Values below are **×10⁻⁶**; the unchanged1e−5 gate corresponds to **10**. At GH81/GH101/GH121/GH161 the maximum is `focal.log_sd` for every solution. The separate last column identifies the GH61 maximum; full signed vectors/labels are retained.

| Slot | GH61 | GH81 | GH101 | GH121 | GH161 | GH61 max coordinate |
|---|---:|---:|---:|---:|---:|---|
|P09.H0|0.057207|9.284767|8.978726|8.974262|8.976379|P04.shared.log_gap_b2|
|P09.Hb|0.051876|9.333500|9.042670|9.037876|9.040194|P01.shared.b1|
|P09.Ha|0.054632|10.852091|10.507139|10.505110|10.507794|P09.shared.log_gap_b3|
|P09.H2|0.288770|11.996627|11.637317|11.634790|11.639017|P05.shared.log_gap_b2|
|P10.H0|0.055325|13.217839|12.785167|12.782858|12.787477|P06.shared.log_gap_b3|
|P10.Hb|0.057347|12.345472|11.955759|11.952557|11.957158|P02.shared.log_gap_b2|
|P10.Ha|0.043165|12.617098|12.218330|12.216523|12.220892|P03.shared.log_gap_b2|
|P10.H2|0.288770|11.996627|11.637317|11.634790|11.639017|P05.shared.log_gap_b2|

**Six of eight slots (five of seven unique solutions) still exceed the native gate at GH161.** P09.H0/Hb were below the gate already. The worst case is P10.H0: GH81=1.32178385e−5; GH161=1.27874769e−5, still27.9% above1e−5. GH81 finite-rule error contributes to the difference but cannot explain away the persisting dense-grid residual.

## Log-likelihood sensitivity at identical parameters

Absolute GH61 logLik and changes relative to GH61 (no refit). Full absolute values and GH161-relative differences are in the machine-readable diagnostic.

| Slot | logLik GH61 | Δ GH81 | Δ GH101 | Δ GH121 | Δ GH161 |
|---|---:|---:|---:|---:|---:|
|P09.H0|-11605.226760846|0.0003474751|0.0003297228|0.0003301735|0.0003302017|
|P09.Hb|-11604.602579961|0.0003468893|0.0003295252|0.0003299898|0.0003300230|
|P09.Ha|-11604.379226187|0.0004158655|0.0003965536|0.0003972536|0.0003972786|
|P09.H2|-11602.984438339|0.0004573610|0.0004379697|0.0004388877|0.0004389374|
|P10.H0|-11606.813502770|0.0005006910|0.0004789562|0.0004799564|0.0004800181|
|P10.Hb|-11603.125816484|0.0004908619|0.0004695598|0.0004705668|0.0004706199|
|P10.Ha|-11606.551578189|0.0004615990|0.0004417524|0.0004426916|0.0004427484|
|P10.H2|-11602.984438339|0.0004573610|0.0004379697|0.0004388877|0.0004389374|

Max GH121–GH161 logLik difference **6.17e−8**; max normalized gradient-vector change **4.62e−9**, far smaller than the residual above the gate. The sequence is not strictly monotonic. These are observed finite-grid differences, **not rigorous infinite-quadrature error bounds**. No tail interval was explicitly truncated; the diagnosis is finite-order Gaussian integration, not proof of a tail-cutoff bug.

## Independent likelihood, derivatives and parameter scaling

- Independent reference: direct respondent log-likelihood with stable category CDF differences, no native pattern/score code. SciPy probabilists `roots_hermitenorm` is normalized to N(0,1), independently checking native NumPy physicists hermgauss×sqrt(2). Max native/reference logLik discrepancy **5.33e−12**; node discrepancy≤1.20e−14; variance-moment discrepancy≤4.44e−16. No normalization/sign evidence was found.
- Central differences cover **all334 free coordinates** of seven unique solutions at GH61 andGH161, using h=1e−4 and1e−5 times max(1,abs(p)). Every perturbed point remains within native bounds. At GH161 the maximum analytic/FD discrepancy is **2.44e−9** for the larger step and **2.25e−10** for the smaller step; GH61 discrepancy≤2.46e−9. These arithmetic checks support the analytic gradient; no scientific acceptance tolerance was changed.
- Chain-rule sensitivities in physical thresholds/SD are also preserved. For P10.H0, saved focalSD=**1.10576679**; GH161 gradient is **−1.27874769e−5 in logSD**, **−1.15643524e−5 in SD**. Their ratio is exactly the SD Jacobian. Replacing the gate with a differently scaled coordinate would change its meaning and is not done.
- For P10.H0, total logLik score in logSD is **+.01278748**. Independent GH161 likelihood with fixed other parameters changes by **+9.70974e−8** at logSD+1e−5 and **−1.58654e−7** at logSD−1e−5. At ±1e−4 both changes are negative (−1.79912e−6/−4.35676e−6), showing curvature matters; this is a local sensitivity diagnostic, not an optimization or estimated optimum. All-coordinate perturbation changes are archived.

## Interpretation and permitted next step

GH61 gradients are near zero at the saved GH61 solutions; denser-grid gradients stabilize away from zero, dominated by the focal variance nuisance direction. Thus a solution stationary for the approximate GH61 objective need not be stationary for a more accurate marginal likelihood. Denser **evaluation alone** does not repair the saved solution. A separately preregistered denser-grid optimization/validation may be needed if the goal is to obtain a higher-order stationary estimate; its success, p values and effect on FWER cannot be inferred here. Selectively replacing this failed replicate would violate PR123’s original design.

**No evidence justifies relaxing1e−5.** Keep the failed record and p-value suppression. No gradient defect, simple factorN/scaling error or numerical instability is supported by these checks; this does not certify exact integration, parameter recovery, global optimality or all GRM implementations. No new p values, Wald intervals, Hessians or scientific model labels were produced.

## Files, execution and tests

`diagnostic.py` executes once from the repository root with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m validation.gh81_diagnostic.diagnostic`. It refuses to overwrite the canonical result and installs runtime guards forbidding native optimizer/fit calls. Actual execution: **10.19 seconds,0 fits**. `results/diagnostic.json` contains full parameter vectors, shared solution mapping, gradients, physical sensitivities, finite differences, versions, hashes and input provenance; `results/execution.txt` is the compact stdout record.

**4 distinct targeted tests passed**: independent likelihood/gradient on a tiny deterministic fixture, physical Jacobian/normal moments, archived reconstruction/aliasing, and cached full-diagnostic integrity/FD checks. New CI tests use only tiny fixtures and stored evidence, never this full calculation or a calibration; existing mandatory checks are retained. No full regression was launched locally. PR123 and all prior canonical records are unchanged. Do not merge automatically.
