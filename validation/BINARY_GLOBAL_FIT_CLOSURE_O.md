# Binary IRT global-fit closure — 2026-11-26

Prospective protocol commit: `521be0af635198e7bc0ab326b47e0ab7a066ca86`.
Independent method selection: `1d166c5310c34136df51f8e58fb09630206512e9`.
Canonical actual checkout: `f3d8d24d76c45e48b5836ba057a2c1d4aab4e441` (GitHub PR merge of that head into unchanged main69be9aa).
Actual confirmation: [run45856527113](https://github.com/Tatung40/psikoanaliz/actions/runs/45856527113).
Prior22 relevant output groups preserved:14 P1 +6 P2 +3PL S-X² constrained reference +4PL separateQ. New groups: **2**, giving **24**; three individually qualified detection-class claims are not three extra output groups.

## Identification and mathematical references
Native Python mirt1.2.0 public `compute_m2`/`diagnostics.modelfit.compute_fit_indices`; no new estimator. Fixed latent N(0,1), D=1, full first/pair covariance through fourth-order unions, full physical/equality-constrained Jacobian. Fixed-a native Rasch/1PL differs from free-common-a1PL; local sigma-free `core.rasch` remains unchanged.

Actual Rmirt1.48/R4.6.2, no-fit ProbTrace/M2, identical GH61 nodes AND weights. Independent R finite differences at1e-5/5e-6 and union covariance agree: maximum covariance error2.22e-16, physical-Jacobian error8.62e-12, projected-M2 difference5.29e-8. Native R M2 agrees for fixed/common1PL and2PL. R3PL/4PL native dP does **not** match independent finite differences of ProbTrace (maximum normalized shape errors .99592/.99407); independently reconstructing its actual derivative/QR path reproduces its M2 within2.56e-13. Those native results are NOT COMPARABLE, never silently declared agreement. At the4PL fixed point native R df30 versus native Python effective-rank df32. R’s approximate fitted independence baseline and nominal rank are not substituted for the exact same baseline/rank; matched CFI/TLI error≤1.63e-10. Original failed preflights remain in Actions and amendments A1/A2; no statistical acceptance threshold changed.

Native tangent rule1e-6 retained; rank audits at1e-5/1e-7 reported. Rank instability blocks ordinary ML chi-square. Saturated moments have no residualdf: M2/RMSEA/CFI/TLI become NA, while SRMSR/correlation residuals remain descriptive. Fixed-latent scale, convergence, sample identity, covariance condition, box constraints and KKT gates fail closed.

## Prospective design and decision
Native cold MAP/GH61/max500/tol1e-4/default priors; N3000/J12/reference item truth, locally independent N(0,1) null. Independent seeds were disjoint from pilot, rooted202611265201/5202/5203. Pilot100 null +100 perclass/model was development only; no pilot certificate.
Final6000 null +600 per each of3 alternatives/model: **15600 actual independent calibrations**, all converged, no refusals. No full-refit bootstrap claim. Joint score/moment Gaussian references retain MAP prior gradient/curvature, full nuisance dependence, native boxes and nonzero means. 3PL uses linear and4PL nonlinear moment re-evaluation; both preserve the **unchanged native M2 statistic**. Monte Carlo reference uses4095 draws plus independent1024 training draws; plus-one tails do not make the fitted Gaussian approximation an exact test.

Preregistered simultaneous null upper CP alpha.025/model≤.07, lower≥.03, dataset-cluster MCSE≤.004, refusal≤.01, convergence≥.99. Each of6 class/model power claims uses one-sided CP alpha.025/6, lower>.10. Null eligible final event-count range207–381/6000; passing probability at true .05=.99999832/model (prospectively calculated before fits). No repeated interim acceptance looks.

| Model/reference | Null events/R | Type-I | simultaneous CP | MCSE | Refusal | reference median/P95 seconds |
|---|---:|---:|---|---:|---:|---|
|3PL native M2 / constrained linear MAP|270/6000|.045000|[.039894,.050554]|.002676|0|.165970/.220020|
|4PL native M2 / constrained nonlinear MAP|260/6000|.043333|[.038321,.048795]|.002629|0|.191375/.285319|

CP endpoints above are marginal97.5% one-sided endpoints per model; the upper bounds jointly protect both model size claims at95%. They are **not** a joint95% interval for every null and power comparison. Registered power endpoints separately protect6 lower-bound claims with family error.025. This certificate is liberal size≤.07, not uniform exact nominal .05 control.

|Model|Registered alternative|Events/600|Power|simultaneous lower|Decision|
|---|---|---:|---:|---:|---|
|3PL|item1 nonmonotone quadratic component -.25|29|.048333|.028219|NOT CERTIFIED|
|3PL|item1–item2 conditional dependence coefficient1|561|.935000|.903832|SCOPED CERTIFIED|
|3PL|factor loading .7 on items1..3|155|.258333|.212538|SCOPED CERTIFIED|
|4PL|same nonmonotone alternative|26|.043333|.024424|NOT CERTIFIED|
|4PL|same dependence alternative|159|.265000|.218761|SCOPED CERTIFIED|
|4PL|same second-factor alternative|25|.041667|.023176|NOT CERTIFIED|

No effects, affected items or gates were changed after pilot. Weak global sensitivity is not structural impossibility; already-certified4PL itemQ sensitivity is preserved. Limited first/pair moments can miss departures absorbed by nuisance/asymptote fits; global and item-specific tests target different departures.

## Production and outputs
M2, effective df, descriptive N−1 RMSEA2, SRMSR, exact independence-baseline CFI/TLI and correlation residuals enter cached summaries, CSV/XLSX/ZIP and Word capture. Ordinary3PL/4PL M2 χ² and RMSEA noncentral-χ² intervals remain withheld. Native undefined-baseline perfect-fit fallback is replaced by NA/reason; TLI is unbounded, conventional CFI truncation explicit. Separate QLI pilot was not selected and is never substituted into M2 indices.

Scoped global reference p requires exact N3000/J12/cold-single-fit/default MAP/GH61/max500/tol1e-4 profile plus frozen-source/registry checks. No implicit cold-start claim: multi-start, unknown profile, other N/J/settings, 4PL D4 solver profile50000/1e-10, missing/changed proof all withhold new inference. The user’s unknown DGP cannot be certified by dimension checks; scope/failed alternatives are shown. Finite-sample certificate is for the registered reference truth, not generic latent distributions or parameter values, no itemwise FWER/FDR.

Regular interior, fixed-scale ML fixed/common1PL→2PL can expose explicitly asymptotic nested LRT, only same complete cohort, scale/quadrature/settings, full rank and no active bounds. MAP and c=0/d=1 boundary comparisons never receive ordinary Wilks p. Unresolved4PL ML maxima are excluded from classical comparison recommendations. LogLik/AIC/BIC descriptions and all old model algorithms remain. No finite-sample LRT certificate is claimed.

4PL C(α) c/d score sets already computed on explicit opt-in are now retained in XLSX/ZIP alongside CSV, separately named; they are never Wald intervals. No inversion is triggered merely by downloading a report. Data/coefficients/settings/proof/seed/profile changes invalidate diagnostics/exports without refitting; unchanged reruns retain cached bytes. Reproducibility bundles include frozen runtime dependencies and proof files.

## Evidence, audit and remaining limits
`evidence/binary_global_fit/CONFIRMATION.jsonl.gz` keeps all attempted repeat decisions, all4 candidate tails, seeds, hashes, errors and diagnostics; original24 canonical checkpoint artifacts retain exact parameters/counts/full diagnostics. Permanent coefficient archives retain every saved estimate without refits. SHA256/Git blobs and actual software/run provenance accompany evidence. Independent SciPy CP/binomial/MCSE audit reproduces reported decisions from raw repeats, including worst-case failure policy. Final CI is archival-only plus software fixtures, never retries or missing-cache refits.

R numerical agreement, pilot success and CI success are not certification by themselves. No new MAP-centred SE/CI, old4PL c/d Wald, ordinary S-X² χ², person-fit p, RMSEA interval, ML4PL optimum or generic-design claim is made. Previously documented P1/P2/P3 scopes and impossible/low-power alternatives remain. Relevant source/certificate blobs are locked against baseline main. No polytomous/CTT/DIF/MFRM algorithm was changed.
