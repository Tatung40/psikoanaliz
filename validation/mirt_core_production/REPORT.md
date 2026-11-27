# Core MIRT production closure

Baseline: `b5b5ab095d00f1ff19d316a2381efcd551ac03d8`. Prospective ownership/protocol committed before implementation; optional cold-NRM protocol: `af024e73e4be99d0db1a2a0df400039f98ab6095`. Policy D applies. No inference certificate is added and no historical certificate is changed.

## Model and numerical scope

| Model | Dimensions / CFA | Engine | Numerical evidence | Inference | Intended release |
|---|---|---|---|---|---|
| M2PL | 2D pair-correlated between-item; 3D orthogonal with two cross-loaders | native Python mirt1.2.0 ML | Rmirt1.48 runs40383677441 | item SE/CI/p WITHHELD | scoped numerical production |
| MGRM | 2D orthogonal cross-loaders; 3D F1-F2 correlated between-item | same | same | WITHHELD | scoped numerical production |
| MGPCM | 3D full-correlation between-item | same | run40394398251 | WITHHELD | scoped numerical production |
| MPCM | 2D orthogonal fixed unit loading vectors | native GPCM fixed discrimination | run40394398251 | WITHHELD | scoped numerical production |
| Multidimensional Rasch | 2D orthogonal fixed unit vectors | native constrained 2PL | run40394398251 | WITHHELD | scoped numerical production |
| MNRM | 2D full category-by-factor slopes, category0 zero reference | native Python NRM | corrected native R gpcm_mats/BOBYQA warm run40399106868 | WITHHELD | experimental cold; optional C1 pending |

Cold accepted scopes are N400, J10(2D)/14(3D), C2 binary or C4 categorical, GH15/axis, max1000/tol1e-7, ML/no priors, zero means/unit variances. Exact masks/correlation/free coordinates are read from existing evidence JSON, not inferred from model name. Input labels/IDs remain user data, not performance score assumptions. Program checks observable scope, packageversion, native masks/bounds/covariance/convergence; latent normality/local independence are assumptions not verified directly from user data. Unknown/evidence-withdrawn/out-of-scope cases remain experimental.

| Profile | N/J/q | slope Δ | transformed d Δ | probability Δ | IIF matrix Δ | EAP Δ | posterior SD Δ | common LL Δ |
|---|---|---|---|---|---|---|---|---|
| GRM_3D | 400/14/3 | 2.1818e-4 | 8.9677e-5 | 5.1512e-5 | 1.9733e-4 | 1.1937e-4 | 1.7956e-5 | 6.8793e-7 |
| GRM_2D | 400/10/2 | 6.4653e-5 | 6.8878e-5 | 2.2612e-5 | 4.4158e-5 | 4.5902e-5 | 5.3368e-6 | 3.2574e-7 |
| 2PL_2D | 400/10/2 | 1.3599e-4 | 7.8175e-6 | 1.1448e-5 | 1.0470e-4 | 5.6559e-5 | 5.5603e-6 | 1.0653e-7 |
| 2PL_3D | 400/14/3 | 2.0457e-4 | 1.0538e-5 | 1.3157e-5 | 1.7881e-4 | 7.5014e-5 | 1.4102e-5 | 1.6228e-7 |
| Rasch_2D | 400/10/2 | 0.0000e+0 | 2.8382e-5 | 6.4517e-6 | 2.7303e-6 | 9.4488e-6 | 7.5305e-7 | 4.7839e-8 |
| PCM_2D | 400/10/2 | 0.0000e+0 | 4.5832e-5 | 1.2079e-5 | 5.4779e-6 | 1.6979e-5 | 1.3044e-6 | 1.3203e-7 |
| GPCM_3D | 400/14/3 | 1.7713e-4 | 1.7629e-4 | 2.6220e-5 | 2.1908e-4 | 5.6042e-5 | 1.0580e-5 | 2.1734e-7 |

Threshold/step comparisons are via d=-sum(a)*b/step; those errors are not direct raw-threshold errors. All seven original independent pairs converged and passed the unchanged prospectively registered criteria. EAP/SD R comparison only first16 fixture persons; P/I evaluated on [-2,0,2]^q. Finite-grid identity is not quadrature accuracy or global optimality.

## Output-level decision

| Output | Evidence / status | Restriction |
|---|---|---|
| Discrimination vectors, intercepts, category thresholds/steps | scoped independent numerical comparison | no universal multidimensional scalar difficulty; raw IRT slopes are not standardized FA loadings |
| Latent covariance/correlation | archived independent numerical comparison / covariance normalization identity | zero means/unit variances and exact declared free correlations |
| EAP / posterior SD | scoped numerical, conditional | no frequentist coverage or new-person/in-sample extrapolation from these fixtures |
| Full posterior covariance | public posterior weighted-node identity; algebraic tests | no independent R covariance or coverage certificate; explicit memory/API refusal |
| MAP person modes | existing conditional calculation, experimental comparison scope | no independent R MAP-person/coverage claim |
| Category probabilities, response surfaces | scoped finite-grid comparison + formula/normalization checks | named rays/slices, other coordinates zero |
| Item/test information | full native matrices; independent R IIF, exact summation | directional e'Ie; no arbitrary universal scalar |
| Directional / joint Fisher uncertainty | quadratic forms and stable full inverse | fixed-orthogonal direction1/sqrt(e'Ie) differs from sqrt(e'I^-1e); singular inverse withheld |
| logLik/AIC/BIC/npar | matched finite-grid ML and arithmetic counts | descriptive, no model selection/Wilks extrapolation |
| Item parameter SE/CI/p | WITHHELD | no observed-information/coverage certificate |
| Global/item-fit inference, residual p, Wilks comparison | WITHHELD | no calibrated multidimensional reference |
| Raw coefficients / matrices / plots / exports | software parity only | scientific status and warnings travel with output; not new inference certificate |

## Iteration ledger

| Mechanism | Hypothesis / correction | Evidence / decision |
|---|---|---|
| P1/P2 cold accepted profiles | blanket experimental label hid existing implementation-level support | unchanged native source + original R reports; attach exact-scope numerical status, leave inference closed |
| NRM original disagreement | R1.48 nominal category-score gradient omissions | preserve original failure and accepted BOBYQA warm evidence; no cold promotion without C1 |
| Lost archive | old artifact40499970808 has no retrievable artifacts | explicitly audit retained immutable reports and canonical source; never claim a fresh saved-fit reevaluation |
| Shared CSV/XLSX/PNG | warnings did not reliably travel with detached exports | authoritative metadata/scope in all bundles; preserve original numerical columns |
| Scoring/reporting | full covariance and directional definitions must use actual APIs | public ability_posterior moment identity, bounded memory, no calibration; explicit WITHHELD on unsupported API |
| Freshness | data/settings/evidence/selection must not leave stale reports or refit on display changes | session-local content hashes and derived-only freshness; verify actual Word and ZIP cache |

## Evidence and source integrity

Native core fit/report functions, compare.py/reference.R/criteria.json are byte-identical to canonical452db8fc6dde0028964c998064020f6713d20c24 apart from a warning constant. OWNERSHIP.json locks the actual baseline scientific/DIF/unidimensional sources and historical evidence; no old hash is rewritten. Existing NON_DIF_IRT_CLOSURE.json is a historical snapshot, not silently promoted. Current numerical scope is a view of existing reports in the existing runtime status system; no competing certification registry.

Existing advanced functionality is retained with its restrictions. No EFA/bifactor/3PL/4PL estimator or optional R backend is implemented. Automatic PR validation audits frozen evidence and mathematical contracts; advanced independent calibration remains explicit manual dispatch only, so old campaigns are not repeated for shared UI edits.

## Software checks and artifacts

Focused API/mathematical/scope/refusal tests, actual native software fixtures, actual CSV/XLSX/ZIP/PNG/DOCX files, real Streamlit no-refit/staleness flow and offline runtime-assets checks are required. Software fixtures are not independent calibration campaigns and do not establish new scientific inference. Actual runtime measurements distinguish calibration and reporting and are single-fixture observations, not unsupported P95/speed claims. Test/CI status remains PENDING until actual Actions completion; raw logs/artifacts are uploaded even on failure.

## Claude integration handover

Shared contract remains result={metadata:dict,tables:dict[str,DataFrame]}. Metadata Status/Numerical_status/Inferential_status and Limitations/Applicable_conditions/Unverifiable_assumptions are authoritative. Reports preserve unknown advanced tables and all original numbers; extra metadata does not grant an advanced certificate. New core helpers apply only owned native CFA; EFA/bifactor/3PL/4PL paths remain as before.

Public interfaces: irt_mirt_policy.profiles/numerical_scope/evidence_fingerprint; irt_mirt_diagnostics.enrich; irt_mirt_production.decorate_native/refresh_policy; shared report.export_fingerprint/downloads/figure. Advanced integrations should provide actual convergence/model/engine/identification/settings/status and reuse the result contract, not copy core numerical status. Add advanced-specific evidence only on Claude's branch after rebase onto merged core main. Keep unsupported item SE/CI/p and generic difficulty closed. Final main/PR/test identifiers are in the PR closure record.
