# DIF method × output certification matrix — Policy D (hybrid), final 2026-11-27

Levels: **NUMERICALLY CERTIFIED** (pre-registered agreement with an external R package) · **INFERENTIALLY CERTIFIED — SCOPED** (pre-registered size/FWER/power gates passed in a stated domain) · **VALIDATED WITH LIMITATIONS** (numerical agreement + documented inferential limitation; screening use) · **EXPERIMENTAL** · **WITHHELD** (not displayed/closed).
No DIF output is INFERENTIALLY CERTIFIED today.

| Method | Output | Level | Evidence |
|---|---|---|---|
| Binary MH | χ²_MH (cc .5), α_MH, ETS Δ, p, BH p, purified χ² | NUMERICALLY CERTIFIED | difR 6.1.0, 2 complete fixtures, ≤2.8e−13 (A) |
| Binary MH | lnOR 95% CI | VALIDATED WITH LIMITATIONS | own R formula only (C) |
| Binary MH | p for DIF decisions | VALIDATED WITH LIMITATIONS — screening only | archived MC R=100 (B) |
| Binary logistic | LR omnibus, LR uniform, p, Nagelkerke ΔR² | NUMERICALLY CERTIFIED | difR difLogistic ≤9.0e−13 (A) |
| Binary logistic | LR non-uniform, Jodoin–Gierl class | VALIDATED WITH LIMITATIONS | own R code (C) |
| Binary logistic | p for DIF decisions | VALIDATED WITH LIMITATIONS — screening only | Type I .073 under impact (B); .062–.067 2PL+impact (E) |
| Ordinal Mantel | χ², SMD | VALIDATED WITH LIMITATIONS | own R code (C); difR has no ordinal Mantel |
| Ordinal logistic | LR, ΔR² | VALIDATED WITH LIMITATIONS | MASS::polr (C); **no difR ordinal LR exists** (difR is dichotomous); lordif uses IRT matching → not comparable — gap |
| Ordinal Mantel / LR | p for DIF decisions | VALIDATED WITH LIMITATIONS — screening only | null Type I .062–.074 under impact (B) |
| SIBTEST (original) | β, χ², p | NUMERICALLY CERTIFIED as Python mirt = R mirt 1.48 | same algorithm, two implementations — not an independent algorithm (A) |
| SIBTEST crossing, Raju areas | β_cross, signed/unsigned area | EXPERIMENTAL | analytic checks only; no R comparison |
| IRT-LR 2PL, Lord χ²/Wald 2PL | statistics, contrasts, covariance | NUMERICALLY CERTIFIED vs mirt (prior PRs) | DIF_CENTER.md, DIF_WALD_LORD_REFERENCE.json |
| IRT-LR / Lord / Wald | p for DIF decisions | VALIDATED WITH LIMITATIONS — screening only | MC R=40, no cluster MCSE (B) |
| Lord 1PL fixed-SD | p / inference | WITHHELD (descriptive only) | DIF_CENTER.md |
| Rasch free-SD Lord | statistics | VALIDATED WITH LIMITATIONS (gated) | DIF_CENTER.md |
| 3PL/4PL bootstrap LR | bootstrap p | EXPERIMENTAL | one design; null 3PL .045 [.027,.071] (B) |
| GRM / polytomous DIF | contrasts, curves | VALIDATED WITH LIMITATIONS (30 paired seeds) | DIF_CENTER.md; FWER not certified |

## Evidence classes and corrections
- **(A) Archived CI evidence** `evidence/dif_difr/COMPARISON_v2.json`: **23 substantive comparisons passed + 3 empty SIBTEST comparisons** (`dif2pl_none` has every item as anchor, no suspect item) — not 26 substantive passes. `dif1pl_uniform` excluded (1174 missing responses; observed-score DIF refuses missing data).
- **(B) Archived MC** `evidence/p2_dif/dif_validation_study.json`: R=40–100 datasets, item-pooled CIs, no dataset-cluster MCSE → cannot certify. The **FWER .24** figure is exactly: ordinal MH, *DIF* condition (FWER among truly null items), N 500/500, μ_F=−.5, BH (`summary` row "ordinal MH DIF −0.5"); null-condition ordinal MH FWER_BH is .09 (μ_F=−.5) and .07 (μ_F=0).
- **(C)** own R re-implementation (`validation/dif_classical_r`), independent code but not an external package.
- **(E) Exploratory only** `evidence/dif_impact_explore/EXPLORE.json` (R=100/cell, seed 20261127041, cluster MCSE .005–.008); diagnostic, not confirmatory.
- Confirmatory 36×500 campaign: **not launched** (Policy D); design draft retained, unregistered.

## Production safeguards
`core.dif_classical.OBSERVED_SCORE_NOTICE` on the observed-score page and in the XLSX Meta sheet; `SCREENING_STATUS` column on every DIF row (screen → DOCX capture, CSV, XLSX); SIBTEST limitation in `core.dif_center.LIMITATIONS` (regression-corrected anchor matching, Python=R mirt, screening p). Existing suppression (purification failure, missing data, fixed-SD Lord p) unchanged. No effect-size cutoff added.
