# Non-DIF IRT final closure audit

Baseline main: `1f4460e6a89e997c18e235382699437a0413772c` (PR207). Existing PR205 is reused; no duplicate PR and no new scientific campaign.

**Full scientific certification: NO.** Certificates attach to an estimator/output/design, never an entire model. Safe scoped production is conditional on required CI and fail-closed unsupported inference, not evidence of full certification.

## Status definitions and counts

C = CERTIFIED scoped claim (including calculation identities explicitly identified below); P = PARTIAL numerical or incomplete evidence; N = NOT CERTIFIED. A withheld output is N, never C. 34 implementation/estimator profiles × 10 output categories: **87 C / 161 P / 92 N**, before and after. Grouping is an audit index, not 340 independent experiments. Two new global-fit claims expand already partly certified item/model-fit cells; they do not promote a whole-model cell.

Named relevant inference groups: **29 → 31 = 14 P1 + 6 P2 + 2 P3 + 7 P4-C1 + 2 global M2 references**. These are distinct from matrix-cell counts and not a universal count of all historical numerical certificates.

| Model/profile | Item | SE | CI | Theta | Theta uncertainty | Probability/info | Descriptive fit | Inferential fit | Comparison | Residual/LD |
|---|---|---|---|---|---|---|---|---|---|---|
|Local Rasch (free sigma)|P|P|P|P|N|C|P|P|P|P|
|Native Rasch / fixed-a 1PL|C|C|C|C|P|C|P|P|P|P|
|Common-a 1PL|C|C|C|C|P|C|P|P|P|P|
|2PL ML|C|C|C|C|P|C|P|P|P|P|
|3PL ML|P|N|N|P|N|C|P|N|P|P|
|3PL MAP|C|C|C|C|C|C|P|C|P|P|
|4PL ML|P|N|N|P|N|C|P|N|P|P|
|4PL MAP|C|C|C|C|C|C|P|C|P|P|
|GRM|C|C|C|C|P|C|P|N|P|P|
|PCM|C|C|C|C|P|C|P|N|P|P|
|GPCM|C|C|C|C|P|C|P|N|P|P|
|NRM|C|C|C|C|P|C|P|N|P|P|
|RSM|P|P|P|P|P|C|P|N|P|P|
|MIRT Rasch/unit-a|P|N|N|P|N|P|P|N|N|P|
|MIRT 2PL|P|N|N|P|N|P|P|N|N|P|
|MIRT GRM|P|N|N|P|N|P|P|N|N|P|
|MIRT PCM|P|N|N|P|N|P|P|N|N|P|
|MIRT GPCM|P|N|N|P|N|P|P|N|N|P|
|MIRT NRM|P|N|N|P|N|P|P|N|N|P|
|MIRT bifactor|P|N|N|P|N|P|P|N|N|P|
|MIRT fixed-c 3PL|P|N|N|P|N|P|P|N|N|P|
|MIRT fixed-c/u 4PL|P|N|N|P|N|P|P|N|N|P|
|MIRT free-c 3PL|P|N|N|P|N|P|P|N|N|P|
|MIRT free-c/u 4PL|P|N|N|P|N|P|P|N|N|P|
|MIRT shared-step RSM|P|N|N|P|N|P|P|N|N|P|
|MFRM TAM|C|P|P|P|P|P|C|N|N|P|
|RaschPy global|C|P|P|C|P|C|C|N|N|P|
|RaschPy items|C|P|P|C|P|C|C|N|N|P|
|RaschPy thresholds|C|P|P|C|P|C|C|N|N|P|
|RaschPy bivector|C|P|P|C|P|C|C|N|N|P|
|RaschPy matrix|C|P|P|C|P|C|C|N|N|P|
|RaschPy centrality|C|P|P|C|P|C|C|N|N|P|
|RaschPy pseudo_halo|C|P|P|C|P|C|C|N|N|P|
|RaschPy bistretch|C|P|P|C|P|C|C|N|N|P|

The companion [machine-readable matrix](NON_DIF_IRT_CLOSURE.json) supplies exact scope, sources and qualifications for every row. C in probability/information denotes a calculation identity, not recovery or interval coverage. RaschPy and TAM C outputs retain their original fixture/recovery meanings; numerical agreement does not establish uncertainty calibration.

## Evidence reused and production correction

* PR207 restored the certified adapter and q1_q3 manifest; all seven P4-C1 reports/protocols remain immutable. P4 SE/CI outside its exact registered design, failed evidence, invalid SE or nonconvergence now remain NA in screen/CSV/XLSX/ZIP/Word. The same native SE method is inspected; NRM explicitly requires Oakes. Evidence/content hashes invalidate derived report caches without refitting. This is a refusal correction, not a new uncertainty method.
* Existing PR205 canonical Actions run **45856527113** performed **15600** calibrations: per model 6000 null and 600 per each of three registered alternatives. Frozen ledger, source hashes, estimates and failures are retained; archival CI only audits them. No new simulations are needed.
* 3PL unchanged M2 + constrained linear Gaussian MAP reference: Type-I **.045**, cluster MCSE **.002676285**, simultaneous CP **[.039893628,.050554404]**; LD power **.935**, second-factor **.258333**. Nonmonotone power **.048333** fails.
* 4PL unchanged M2 + constrained nonlinear reference: Type-I **.043333**, MCSE **.002628547**, CP **[.038321340,.048795165]**; LD power **.265**. Nonmonotone **.043333** and second-factor **.041667** fail. Refusal zero both. These satisfy only their registered liberal size/power gates; they do not establish arbitrary nominal control or sensitivity.
* Post-fit reference P95 **.220/.285 s** (3PL/4PL). Only reference truth, N3000/J12, normal theta, single cold native MAP/default priors/GH61/max500/tol1e-4 is certified. Other DGPs/settings, D4 or multistart do not inherit it.
* Existing 4PL c/d **C(alpha) score-inversion sets** remain the bounded supported solution. Preserve weak/uninformative/empty-set diagnostics; ordinary c/d Wald intervals remain withdrawn. No further inversion/calibration is scientifically necessary for this closure.

## Independent R and remaining scientific limits

Archived actual R mirt 1.48 point/curve/information comparisons cover GRM/PCM/GPCM/NRM, binary matched fixtures and RSM mappings with recorded identification/quadrature limitations. P4-C1 uncertainty evidence is the separate preregistered 400-calibration/model study, not the R comparison. MIRT numerical R agreement is explicitly not inferential certification. TAM/RaschPy retain independent native references.

For 3PL/4PL native R M2 is **NOT COMPARABLE** (MAP nuisance projection, rank and degrees of freedom differ); independently implemented R moment/Jacobian mathematics agrees to **5.95e-8**. Do not replace that qualified result with R equivalence. Rasch/common-a 1PL/2PL ML comparisons retain their recorded numerical scope.

Unsupported native MAP Wald inference, ordinary MAP M2/S-X2 chi-square, RMSEA confidence intervals, generic boundary Wilks tests, person conditional credible-tail coverage, in-sample/J30 extrapolation and broad MIRT/RSM/MFRM uncertainty remain N/P as specified. Existing 3PL S-X2 constrained reference and separately named 4PL Q are preserved, never relabelled ordinary chi-square. RSM LR null evidence (R100, estimate .07, CP [.0286,.1389]) is insufficient to certify error control. Bifactor N500 recovery fails; free-asymptote MIRT is limited to accepted profiles.

Plots/export parity are software contracts: they cannot certify the scientific quantity they display. Nominal categories are not ordered achievement; NRM Wright/S-X2 unsupported paths remain disabled. B14 and legacy aliases do not count as independent models. No DIF file, method, evidence or gate is changed.

## Verification

Required current-head CI includes the unchanged graphic-selection/no-refit regression, P4 native fixture/table/Word/export/refusal tests, existing binary production/global diagnostics contracts, frozen-evidence audit, full required repository checks and numerical browser comparisons. Software fixtures are not rerun scientific studies. CI results belong to the actual PR head; pending checks are never PASS. No redundant full local regression is run.
