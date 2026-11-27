# DIF method × output certification matrix (audit phase 1, 2026-11-27)

Levels: **NV** numerical agreement with an independent reference · **MC-size** preregistered Monte Carlo size evidence (not certification unless gates + precision met) · **NOT CERTIFIED**. No DIF output is inferentially CERTIFIED (Type I/FWER/power) today.

| Method (module) | Statistic / effect | p / adjusted p | Type I / FWER | Power | Independent reference |
|---|---|---|---|---|---|
| Mantel–Haenszel binary (`dif_classical.mantel_haenszel`) | χ²_MH (cc .5), α_MH, ETS Δ, lnOR CI: **NV** (own R code; difR 6.1.0 χ² 2.8e−13, α 3.1e−15, p 3.4e−15) | **NV** | MC-size R=100: Type I .036–.050, FWER_BH .01–.06 — *not certified* (item-pooled CIs, no cluster MCSE) | not reported | difR::difMH |
| Logistic regression binary (`logistic_dif`) | LR uniform/non-uniform/omnibus, Nagelkerke ΔR², Jodoin–Gierl: **NV** (own R glm; difR LR 6.8e−13) | **NV** | **inflated under impact** μ_F=−.5: Type I .073 (N1000), FWER_BH .07 — NOT CERTIFIED | not reported | difR::difLogistic |
| Mantel ordinal (`mantel_ordinal`) | Mantel χ², SMD: **NV** (own R code) | **NV** | impact μ_F=−.5: Type I .062–.074, FWER_BH .09–.24 — NOT CERTIFIED | not reported | difR has no ordinal Mantel; `DescTools::MantelHaenszelTest`/manual |
| Ordinal logistic (`logistic_dif(ordinal)`) | LR, ΔR²: **NV** (MASS::polr) | **NV** | impact: Type I .074–.080 — NOT CERTIFIED | not reported | lordif (ordinal LR) |
| IRT LR 2PL (`irt_dif`) | LR, contrasts: NV vs mirt (DIF_VALIDATION) | NV | MC-size R=40: Type I .045–.054, FWER_BH .00–.13 — *not certified* (R=40) | not reported | mirt::DIF / multipleGroup |
| Lord χ² / Wald 2PL (`dif_lord`) | χ², parameters, covariance: **NV** vs R/mirt (3 registered designs) | NV | NOT CERTIFIED | — | mirt::multipleGroup (Wald) |
| Lord 1PL fixed-SD | descriptive only (equal latent variance not verified) | **closed** | NOT CERTIFIED | — | — |
| Lord Rasch free-SD (`dif_lord_rasch`) | gated numerical PASS | conditional, gated | NOT FWER CERTIFIED | — | mirt |
| 3PL/4PL parametric-bootstrap LR (`irt_dif34_bootstrap`) | bootstrap LR | bootstrap p | MC-size (warp-speed) 3PL null .045 [.027,.071] — *not certified* (single design) | DIF-M1 rejection .95 | none |
| GRM / polytomous IRT DIF (`grm_dif`, `polytomous_dif`) | contrasts, curves: **NV — 30 paired seeds** | NV | FWER NOT CERTIFIED | — | mirt (multipleGroup), lordif |
| SIBTEST / crossing, Raju area (`dif_center`) | β, β_cross, Raju signed/unsigned: analytic NV (≤2e−14) | NV | NOT CERTIFIED; R (mirt::SIBTEST, difR::difRaju) NOT EXECUTED | — | mirt::SIBTEST, difR::difRaju |
| python-mirt DIF (`python_mirt_dif`) | fit statistics: NV (PYTHON_MIRT_DIF.md) | gated | NOT CERTIFIED | — | mirt |

## Prioritised gaps
1. **Type I inflation under impact** for observed-score LR and ordinal Mantel/LR (μ_F=−.5): a real validity issue; requires either purification/anchor policy evaluation or labelling as impact-sensitive. Highest priority.
2. **No dataset-cluster MCSE / adequate R** in existing DIF MC studies (R=40–100; item-pooled CIs) — any certification needs a preregistered rerun.
3. **Power** never evaluated against a registered criterion for any method.
4. SIBTEST/Raju R references (mirt::SIBTEST, difR::difRaju) not executed.
5. Bootstrap 3PL/4PL DIF: single design; no FWER.
