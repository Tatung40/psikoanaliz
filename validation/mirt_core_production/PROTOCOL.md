# Core MIRT production protocol (prospective software work; no new calibration campaign)

Baseline main: `b5b5ab095d00f1ff19d316a2381efcd551ac03d8`; current main required CI all PASS. Policy D applies. Ownership is recorded in OWNERSHIP.json before implementation. Claude-owned estimators/branches are not changed.

## Fixed evidence and scope

Reuse phase1_evidence.json (R run40383677441) and expansion_original_evidence.json (run40394398251). Native source functions, compare.py, reference.R and criteria.json are identical to canonical commit452db8fc6dde0028964c998064020f6713d20c24 (except a later warning constant). No existing evidence, threshold, hash, adapter, unidimensional or DIF algorithm is changed.

Eligible cold native CFA profiles: 2PL_2D, 2PL_3D, GRM_2D, GRM_3D, GPCM_3D, Rasch_2D, PCM_2D. Every profile must satisfy existing preflight/calibration criteria and actual R/Python convergence. Runtime compares observable N/J/C, loading mask, covariance constraints, quadrature and solver settings to that profile. Latent normality and true DGP cannot be verified from user data and remain explicit assumptions. Unknown/missing/withdrawn evidence, nonconvergence, boundary solutions or profile mismatch cannot receive a scoped numerical certificate. New data are never declared independently revalidated merely by profile matching.

MNRM cold calibration remains experimental: historical R category-score derivative limitation and accepted BOBYQA warm continuation are preserved and reported. No narrowing of scientific acceptance gates; no promotion of exploratory, bifactor, 3PL/4PL or R-backend work.

## Predeclared verification

* Retain original numerical criteria: mapping P1e-10/I1e-8/prior1e-12; parameter/covariance/I/EAP/SD1e-3; probability1e-4; commonLL1e-3/nativeLL1e-6. No new calibration is needed for unchanged accepted scopes.
* Existing native mathematical mapping, model identification, probability/information and negative convergence tests remain mandatory. New posterior covariance is a weighted-node identity through the public installed API, with matching EAP/SD, IDs, finite normalized weights and PSD; not an R-validated covariance or coverage certificate. Directional information must equal e' I e; no arbitrary universal scalar.
* Preserve all existing result tables/numbers and unsupported inference restrictions. New status/limitations must be identical across UI, CSV, XLSX, DOCX/ZIP/PNG and reproducibility manifests; prove with real software fixture outputs and actual parsed export files. No repeated model fitting on display/export changes. Unknown optional API is withheld explicitly.
* Reuse cached fit/report/scores safely by input and evidence content, session isolated. Benchmark actual timings as descriptive measurements; no speed superiority claim or latency threshold tuned after observing it.
* Focused tests first; required CI before merge, no local full suite or new Monte Carlo. Existing advanced scientific calibration steps become explicit manual dispatch only; PR uses frozen evidence/source audit and mathematical checks. Missing expired artifact is reported, never silently replaced by recalibration.

## Loop and stop rules

For each mechanism: hypothesis → minimal correction → targeted test → evidence → scoped decision. At most three substantive attempts and no identical retry. Core models advance independently; optional MNRM does not block accepted scopes. Maximum120 active minutes, CI waiting excluded. No merge with pending/failing mandatory checks. Preserve negative evidence and report unresolved scopes.

## Release interpretation

NUMERICALLY CERTIFIED / SCOPED is an implementation-level finite-profile claim, separate from WITHHELD inferential outputs. Posterior SD/covariance are conditional on fitted items and the finite-grid prior; they are not item SEs or frequentist coverage. ML AIC/BIC are descriptive with native parameter count and matched likelihood; no Wilks/model-fit p. No generic multidimensional scalar difficulty or nominal Wright map is introduced. Shared advanced extension contract is documented without enabling advanced models.

## Prospective optional C1: one cold-start NRM numerical comparison

Before any new cold-reference results, permit exactly one independent pair on the unchanged original NRM_2D fixture: N400/J10/C4, two dimensions, original pattern/full correlation, native unit latent variances/zero means/reference-category zero coefficients. Use original deterministic dataset/seed and no saved/warm starting solution. Both engines start with nominal_start/zero intercepts and identity latent covariance; ML without priors, GH15, max3000 and EM tol1e-9. R uses the already corrected native gpcm_mats and existing BOBYQA reference; native Python public API/production point estimator is unchanged. This fills cold-initialization evidence, not a new population or inferential claim.

Verify fixed-parameter mapping first using unchanged registered preflight gates. Then at most one Python fit and one independent R fit, each bounded to300seconds; no retries/continuations/alternate seeds. Require both convergence/admissibility and every original criteria.json numerical gate unchanged. Save starts, dataset/criteria/source hashes, versions, raw outputs and logs regardless of failure. Timeout, nonconvergence or any unmet gate retains experimental cold NRM; other core releases proceed. Original warm acceptance and original derivative failure remain separate. No new coverage, global-optimum or quadrature-accuracy claim.
