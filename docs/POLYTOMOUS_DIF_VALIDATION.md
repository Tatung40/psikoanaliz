# PCM / GPCM / RSM / NRM: bounded experimental DIF

Base: PR #55, `f63b253`. Existing Rasch/1PL/2PL/GRM sources and fixtures are retained.
All four new families remain **EXPERIMENTAL**, not general FDR/coverage certifications.
GRM's independent R reference and FDR limitations remain open tasks.

## Likelihood, scale and hypotheses

Concurrent two-group unidimensional MML with normal latent distributions;
reference N(0,1), focal mean and SD free. Same respondent subset/category coding
and MAR omission of missing item responses in every nested model. Require >=100
retained persons/group, >=3 unique items, >=2 predeclared invariant anchors and
>=1 target. Each item/group/category cell needs >=5 observations by default
(configurable integer >=2). These are operating bounds, not adequacy theorems.
Identifiers/group columns are excluded by the UI; core rejects a named group column
as an item. Other groups and all-missing respondents are excluded explicitly.

All models use normalized canonical logits eta_c=A_c*theta+D_c, eta_0=0:

| Model | Parameterization and invariance hypotheses |
|---|---|
| PCM | A_c=c, D_c=−sum(b_s,s<=c). Steps may be disordered; never sorted. All C−1 target step locations equal (H0) versus free (H2), LR C−1 df. |
| GPCM | A_c=c*a, D_c=−a*sum(b_s,s<=c), a>0. H0 a,b equal; Hb b free/a equal; Ha a free/b equal; H2 both free. Global H2/H0 C df; conditional step H2/Ha C−1 df; conditional slope H2/Hb 1 df. Physical b equality differs from canonical D equality when a varies. |
| RSM | A_c=c, D_c=−c*location−sum(tau_s,s<=c). The SAME tau applies to EVERY item and group; sum(tau)=0. Target location equal versus free: LR 1 df. No item-specific step freeing and no separate category-step DIF claim. |
| NRM | A_c,D_c freely estimated for each nonreference category. H0 both equal; Hb intercepts free; Ha slopes free; H2 both free. Global 2(C−1) df; conditional intercept and slope tests C−1 df each. Labels are explicitly mapped, never ranked/scored. |

**PCM/RSM scale restriction:** fixing both a=1 and reference variance=1 is an
additional distribution assumption. This is not the existing Rasch model with
estimated reference SD. Swapping groups can change this restricted model;
do not assert a free-slope rescaling symmetry. GPCM/NRM reference-group reversal
is independently tested with the correct coefficient/latent transformation.

NRM fixes A_0=D_0=0. The first predeclared anchor's second/reference category slope
contrast is constrained positive to choose the latent-axis sign. This is not a
success order. A nominal one-hot SVD provides a second numerical start only when
the first fails; no observed matching score enters estimation/testing. Baseline/
label permutation is tested including coefficient recentering and sign reflection.
Changing labels can change proximity to the declared finite bounds; such boundary
fits cannot produce inference.

Unstudied nonanchors remain fully group-specific nuisance items. Anchors are not
tested or selected/purified from these data. Their invariance requires independent
evidence; do not interpret the anchor label as a successful invariance test.

## Numerical gates and uncertainty

Stable log-softmax and analytic marginal score; deterministic bounded L-BFGS-B.
Bounds: GPCM a [.1,4]; PCM/GPCM steps, RSM locations and NRM intercepts [-8,8];
NRM slopes [-4,4] except orientation contrast [.05,4]; RSM free centered-step
coordinates [-4,4]; focal mean [-3,3], SD [.3,3]. The derived last RSM step is the
negative sum of the others, not a separately bounded/fitted coordinate.
Require score/person <2e-6 and all parameters >1e-4 from a bound. Observed
information uses central differences of the analytic score, symmetrized;
min eigenvalue must exceed 1e-7*max(max eigenvalue,1). No conditional submatrix
is substituted for full regular information.

Initial quadrature 61 by default; audit max(Q+20,81). Require absolute total
logL change <=.001 and audit score/person <=1e-5. Failed audits alone trigger
refitting at Q+20 up to 161 (audit <=181); successful fits are cached/reused.
Actual per-fit integration sizes are exported. Nested maxima below the null by
>1e-6 absolute units, nonfinite likelihoods or any failed gate withhold p/intervals;
they do not become p=1 or evidence of no DIF. Different converged maxima use
audited approximations to the same marginal integral.

Separate BH families over nonanchors for global and each conditional component;
no joint-across-family FDR guarantee. Full joint covariance supplies pointwise
parameter/curve delta intervals. Ordinal effects average the matched-theta
expected-score difference over the estimated focal distribution, including its
mean/SD derivatives. NRM exports category probability differences only: no expected
score, scored matching, success/advantage direction or automatic item removal.

## Independent references

Installed **Python mirt 1.2.0 is not R mirt**. Its public PCM/GPCM `steps` are
direct adjacent-step theta locations (not assumed difficulty/centered-step fields).
Public four-category PCM/GPCM/RSM/NRM probabilities agree within 1e-12; analytic
scores agree with independent finite differences; missing-response likelihoods
agree with adaptive normal integration within 2e-7. Weighted-effect derivatives
include the focal moments and match independent finite differences.

Actual independent EM calibrations, seed 20261025, null and planted conditions:
PCM 4 fits; GPCM 8 fits; NRM 8 fits. H0/Hb/Ha/H2 where applicable, N600/400,
6 items/3 categories, four invariant anchors, reference N(0,1), focal N(−.4,1.2²).
Counts match (PCM 16/18; GPCM 23/25/24/26; NRM 30/32/32/34).
Maximum native versus external reported logL differences: PCM <2.5e-5,
GPCM <9.2e-4, NRM <7e-7; canonical coefficient differences <1e-4.
Tests use .002 total logL/parameter tolerances. Recomputed external likelihoods
and scores are saved too: Python mirt's reported GPCM logL differs slightly from
native evaluation at its own coefficients. Do not claim exact equality.

RSM's public multigroup string factory remains unsupported. Instead, FOUR
independent direct respondent-likelihood calibrations implement centered common
steps using 101-point NumPy Hermite integration, physical SD and numerical
derivatives, without native Design/objective/score/compression. Native logL and
parameters agree (<3e-6 coefficient difference). Public RatingScaleModel
probabilities supply another independent formula check. This is NOT an independent
R multigroup benchmark. References are frozen JSON, not recalibrated in CI.

## Cached simulations and limits

Each model: 100 complete-null +100 planted replicates, independent deterministic
SeedSequence streams, source/version/config SHA cache. All 800 final repetitions
usable, zero inconclusive. Code was frozen after fixing GPCM integration audit
failures and NRM start sensitivity; final artifacts identify that exact source.
4,139 optimizer calls (~402 s). No prior Rasch/1PL/2PL/GRM study was rerun.
Rates/coverage condition on valid fits; inconclusive counts are explicit even zero.
Exact Clopper–Pearson MC intervals accompany every rate/coverage in JSON.

| Model | Null: any global BH rejection (95% MC interval) | Planted global detection M5 / M6 | Parameter coverage range | Candidate weighted-effect coverage |
|---|---|---|---|---|
| PCM | 4/100 [.0110,.0993] | 100/100 / 100/100 | .93–.97 | .94–.97 |
| GPCM | 7/100 [.0286,.1389] | 100/100 / 97/100 | .90–.99 | .94–.98 |
| RSM | 5/100 [.0164,.1128] | 100/100 / 100/100 | .90–.97 | .90–.98 |
| NRM | 5/100 [.0164,.1128] | 100/100 / 87/100 | .92–.99 | .88–.98 |

PCM M5 steps +.6, M6 steps −.7; RSM locations +.6/−.7 with genuinely shared steps;
GPCM M5 steps +.6 and M6 a−.6; NRM M5 one category intercept +.9 and M6 one
category slope −.9 (mixed-sign category slopes). GPCM conditional slope detection
70/100 [.6002,.7876], NRM 84/100 [.7532,.9057]; planted intercept/step detection
100/100 [.9638,1]. Global M6 power in NRM 87% is not 100%.

**NRM weighted category-effect intervals undercovered:** minimum 88/100, exact
MC interval excludes .95. Candidate intervals remain in validation artifacts to
show the failure; UI/DOCX/XLSX/CSV withhold these intervals and explain why.
Conditional-curve/parameter intervals remain explicitly experimental normal
approximations, not a general coverage claim. Broader sample sizes, sparse cells,
category counts, anchor contamination, heterogeneous categories and latent
misspecification remain unvalidated. Null MC intervals permit material inflation;
nominal/general FDR control is NOT established for any family.

## Reporting, existing asymptote models and open work

Opt-in separate page; changed data/model/labels/settings invalidate old results.
Turkish DOCX/XLSX/BOM CSV/300-dpi PNG ZIP exports input/missingness/anchors,
coefficients, LR/df/raw p/BH, effects, full optimization/integration diagnostics
and descriptive category margins. Charts use matched reference theta. Formula
labels are escaped, archive paths fixed; no respondent rows/IDs and no refitting.
Model-fit assumptions remain assumptions, not inferred from margin plots.

3PL gains fixed anchors, strict minimums, full-information/integral gates,
pointwise valid-only parameter intervals and withheld inference for unstable final
purification. Existing R calibration fixtures are reused, not relabeled as new
Monte Carlo validation. Exploratory selected-item component p-values remain raw
and explicitly lack separate FDR control. Broad 3PL coverage, probability-effect
intervals and validated boundary inference remain unfinished.
4PL's inferential API is now blocked, matching its existing disabled inferential
UI; only descriptive comparisons remain available. No chi-square boundary or
unexecuted bootstrap claim. General 3PL/4PL completion is NOT asserted.

Bibliographic metadata verified from publisher/author records (2026-10-23):
Masters (1982), DOI 10.1007/BF02296272; Muraki (1992), DOI
10.1177/014662169201600206; Andrich (1978), DOI 10.1007/BF02293814;
Bock (1972), DOI 10.1007/BF02291411. See report references for full citations.
