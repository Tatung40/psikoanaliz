# Scoped certification inventory — 2026-11-05

**Full certification: BLOCKED. No model-wide certification.** Base main
`d9a6266` includes merged #136–#139. This batch corrects evidence attribution,
adds a reproducible claim inventory and independently checks four closed-form
adjustments. It does not resolve every advertised output or restart experiments.

`matrix.jsonl` contains 163 explicit claims: 4 CERTIFIED, 72 PARTIALLY CERTIFIED,
83 NOT CERTIFIED, 4 UNSUPPORTED. These are inventory cells, not 163 models or
an exhaustive denominator for every statistic. Forty current catalog routes
have separate product/export entries. A route entry does not imply that every
listed export format is supported. No model or route is wholly certified.

## New independent execution

`multiplicity.json` records actual Python/statsmodels/NumPy versions, seed,
all 13 inputs, source identities, pre-existing PR #139 tolerance (absolute
1e-10, relative zero) and observed discrepancies. Empty, all-missing, singleton,
zero/one, ties, reversed order and deterministic random fixtures were compared.

| Output (recorded fixtures only) | Max absolute difference |
|---|---:|
| BH adjusted p | 1.1102230246251565e-16 |
| BY adjusted p | 2.7755575615628914e-17 |
| Holm adjusted p | 0 |
| Bonferroni adjusted p | 0 |

CERTIFIED here means numerical implementation agreement **only for these
recorded valid-p fixtures**, including NaN handling. It does not establish
the validity of input p-values, dependence assumptions, or Type-I/FDR/FWER
control for any DIF method. No sampling-inference certificate follows.

## Reused evidence and prioritized blockers

1. **4PL ML:** archived R fit did not converge after 3000 iterations.
   Common-GH61 absolute logLik difference .000228812638, max ICC difference
   .00899980398. Observed-information active-face minimum eigenvalue .0378910,
   scaled condition 2349.497; physical-coordinate weakest eigenvector has
   99.1385% squared loading on M12. Scaling changes eigenvectors: this percentage
   is **not** the diagonally scaled mode. These are local weak-identification
   and optimization findings, not proof of structural nonidentifiability or
   global optimality. Existing information/trace diagnostics were read, not
   recomputed. Keep ML interpretation/CI gates; MAP recovery is separate evidence.
2. **3PL/common-a 1PL inference:** single-fixture agreement or native-engine
   identity cannot certify SE/CI coverage. Reuse registered coverage evidence;
   any extension needs a separate preregistered design, not retrospective limits.
3. **MIRT MAP/bifactor:** existing ML references cannot be transferred to other
   priors, structures, dimensions or inferential outputs. Keep disabled inference.
4. **1PL Lord:** the new separate free-SD Rasch path already has actual independent
   R mirt 1.48 evidence (run41489608564). Same741 complete cases, a=1, mean=0,
   independently free SDs, full nuisance covariance and anchor delta method.
   Max relative chi-square difference 1.671e-5 < registered1e-3; all fits converged.
   This is conditional numerical evidence, not general Type-I/FWER or coverage.
   Legacy fixed-SD results remain descriptive. No new R execution is claimed.
5. **DIF error control:** sequential study completed852 pairs/1704 valid analyses.
   Registered look500: impact−.5 29/500=.058, simultaneous CP[.033928,.091237];
   impact0 25/500=.050, CP[.027860,.081499]. **INCONCLUSIVE / NOT CERTIFIED**.
   Looks1000/1500 were not executed. Descriptive852 prefix cannot become a new
   confirmatory look. Lord/Wald/selected-anchor FWER gaps remain separate.
6. **GT:** balanced p×i ANOVA agreement does not validate fixed facets, unbalanced
   or arbitrary nested designs. Current API restrictions remain unchanged.
7. **NRM:** nominal categories do not define ordinal thresholds. Wright map and
   unsupported ordinal item-fit statistics remain disabled, not missing values
   to manufacture or certification labels to remove.
8. **Polytomous recovery/inference:** analytic probability/Fisher checks conditional
   on coefficients are distinct from independent calibration, SE and CI coverage.
9. **Other routes/product:** existing Word/export/AppTest contracts support software
   consistency. They do not independently validate every displayed statistic.
   The independent same-grid EAP test is a 2PL fixture, not all-model validation.

## Reproduction and integration

No Rscript is installed in this workspace; no packages were downloaded and no
R or Python fits were run. Original R evidence, Monte Carlo archives, protocols,
acceptance limits, production engines and inference gates are unchanged.

Rebuild the **new** records only:

```sh
OPENBLAS_NUM_THREADS=1 python -m validation.certification_registry --write
python -m pytest -q tests/test_certification_registry.py
```

The normal PR pytest selection now includes this evidence gate and the existing
independent-reference tests. A single full regression is requested in this PR
using the existing CI opt-in; its actual outcome is in the PR checks, not
invented in the matrix. Existing required scientific workflows are preserved.

Next highest-impact executable action: a separate, preregistered 4PL optimizer
diagnostic using the saved fixture and exact matched constraints in an actual
R environment. Do not infer convergence from small objective increments or
increase iteration limits blindly. SE/CI and general error-control certification
remain separate work even if numerical optimization succeeds.
