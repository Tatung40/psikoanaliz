# DIF validation — observed-score methods, IRT-LR and 3PL/4PL bootstrap

Builds on `docs/DIF_COMPLETION_MATRIX.md` (fixed-anchor IRT DIF for 1PL/2PL/GRM/PCM/GPCM/RSM/NRM,
200-repetition studies) without repeating it. All designs and acceptance criteria below were written
into the scripts before any result was produced.

## New methods (`core/dif_classical.py`, page **MTK: Gözlenen Puan DIF**)

| Method | Items | Output |
|---|---|---|
| Mantel–Haenszel (Holland & Thayer) | binary | χ²_MH (continuity corrected), α_MH, RBG SE, Wald CI of ln α, ETS D-DIF and A/B/C |
| Mantel (1963) + SMD | ordinal | 1-df χ², focal-weighted standardized mean difference |
| Logistic regression (Swaminathan & Rogers) | binary | uniform (1 df), nonuniform (1 df), omnibus (2 df) LR; Nagelkerke ΔR² with Jodoin–Gierl A/B/C |
| Ordinal logistic regression | ordinal | same nested models with cumulative logits |
| Multiplicity | all | BH, BY, Holm, Bonferroni or none (identical to statsmodels `multipletests`) |

Matching on the total score including the studied item, optional iterative purification. Independent
checks: MH χ², α_MH and RBG SE equal statsmodels `StratifiedTable` to 1e-12; the Mantel statistic
equals the generalized CMH correlation statistic. Group impact (Cohen d of total scores) is reported
and triggers a study-based warning when |d| ≥ .3.

## Type I error, FDR/FWER, power and coverage (`evidence/p2_dif/`)

2PL binary generator, k = 20, four DIF items (two uniform Δb = .5, two nonuniform Δa = −.6) or none;
sizes 500/500 and 1000/200; focal mean 0 or −.5; GRM ordinal generator k = 10 with two threshold-shift
items. 100 replications (IRT-LR 40). 2,560 analyses; one IRT-LR analysis stopped by design
(purification did not stabilize), none hidden.

| Method | Condition | per-item Type I | Holm FWER: complete null / DIF design | uniform / nonuniform power (BH) |
|---|---|---|---|---|
| MH | binary, all four designs | .035–.050 | .01–.06 / .02–.04 | .76–.97 / .18–.41 |
| LR | binary, 500/500 | .043–.062 | .03–.06 / .04–.09 | .86–.95 / .68–.76 |
| LR | binary, 1000/200, impact −.5 | .058–.073 | .07 / .07 | .70 / .60 |
| IRT-LR 2PL (existing `core.irt_dif`) | 500/500 | .045–.054 | .00–.05 / .00–.03 | .85–.89 / .74–.86 |
| Mantel | ordinal, no impact | .055–.059 | .07 / .04 | .79 (threshold shift) |
| Mantel | ordinal, impact −.5 | .062–.074 | **.09 / .12** | .71 |
| Ordinal LR | ordinal, no impact | .049–.054 | .05 / .02 | .75 |
| Ordinal LR | ordinal, impact −.5 | .074–.080 | **.11 / .10** | .68 |

IRT-LR Wald 95 % intervals of Δa and Δb covered the true values in .942–.965 of cases (criterion
[.925, .975] met in all four conditions). Pre-registered per-item Type I criterion [.025, .075] met by
all binary cells and missed by ordinal LR with impact (.080). Family-wise / FDR control is not
achieved by the observed-score ordinal methods under impact: the page warns and recommends IRT-LR.
MH is insensitive to nonuniform DIF (power .18–.41), as expected.

## 3PL/4PL boundary inference: parametric bootstrap (`core/irt_dif34_bootstrap.py`)

The χ² reference is invalid when c/d lie on a bound (in this design ≥ 99 % of fits had a boundary
parameter, so `core.irt_dif34` withholds every asymptotic p-value). The bootstrap simulates from the
fitted null model and refits both nested models; failed replicates are counted and the p-value is
withheld when more than 10 % fail. Size checked with warp-speed Monte Carlo (`evidence/p2_dif34_bootstrap/`;
k = 8, 500 + 500, focal N(−.3, 1.1²)):

| Condition | used | bootstrap rejection [95 % CI] | naive χ² rejection | criterion |
|---|---|---|---|---|
| 3PL null, studied c = .02 | 396/400 | .045 [.027, .071] | .028 | met |
| 3PL null, studied c = .15 | 395/400 | .076 [.052, .107] | .043 | **not met** (> .075) |
| 4PL null, studied c = .02 | 281/300 | .046 [.025, .078] | .018 | met |
| 3PL DIF Δb = .6 (power) | 198/200 | .949 | .889 | — |

Invalid observed fits (non-convergence or nested-likelihood violation) are excluded and counted
(4/400, 5/400, 19/300, 2/200). The committed study's `ui_enabled` field records its historical
model-level criterion only; it is not an input-specific eligibility certificate. Exact eligibility
for arbitrary observed data cannot be established from this restricted warp-speed study. The UI
therefore withholds **both 3PL and 4PL p-values**, including CSV exports, and preserves descriptive
LR statistics and bootstrap distributions. Results are invalidated when data values or fit settings
change, including same-shape replacements. The underlying research bootstrap calculation and the
study's criteria/results remain unchanged.

## Still open

Selection-adjusted inference after data-driven anchor purification; FDR/coverage across broader
designs (more items, unequal slopes in ordinal data, non-normal latent distributions); NRM weighted
category-effect intervals in `core.polytomous_dif` (previous minimum coverage 88 %, still withheld);
3PL bootstrap size slightly liberal for an interior studied item.
