# B12.5 — Independent R mirt cross-validation (uploaded evidence)

> **Arşiv notu (2026-10-29):** Yerel ikili MTK motoru (`core.irt`, `core.irt_mixed`, B10/B11/B12/B13 karşılaştırma altyapısı) kaldırıldı; ikili MTK kestirimi yalnız Python mirt ile yapılır. Bu belgedeki kaldırılan modüllere ait bölümler tarihsel kayıttır. Korunan modüller (`core.irt_mg`, `core.irt_mg34`, `core.irt_dif`, `core.irt_q3`) etkin DIF/yerel uyum ekranlarında kullanılmaya devam eder.

**Status: PARTIALLY VALIDATED — experimental backend only.** Evidence archive: `EserPsychometrics_B12_5_R_Validation.zip` (2026-10-22). This document records supplied outputs; it does not claim that GitHub CI executed R.

## Design
- 5,000 respondents, 30 dichotomous items: M01–M08 fixed-a=1 (1PL), M09–M16 2PL, M17–M24 3PL, M25–M30 4PL.
- Reference group Women: mean 0, SD 1. Focal group Men: estimated mean and SD. Item M05, M12, M20 group-specific, remaining items anchored.
- 61 equally spaced quadrature points on [-6,6], D=1, bounded a=[.15,4], b=[-4,4], c=[0,.35], d=[.65,1] in Python. **mirt b bound is not equivalent**; g and u logit transforms cannot attain exact 0/1.
- Python `core.irt_mixed` direct bounded ML / L-BFGS-B (not EM); R `mirt` marginal ML / EM.

## Main sample numerical evidence (provided CSV files)
- Python logLik **-84547.661509** (80 parameters); boxed R mirt default **-84547.665041**; difference **+0.003532**.
- Python AIC **169255.323017**, BIC **169776.698473**; R AIC **169255.330083**, BIC **169776.705538**.
- Focal latent mean **-0.294938** (Python) / **-0.294940** (R); SD **1.059445** / **1.059448**.
- Independent cross-evaluation of R's four fitted parameter solutions in the Python likelihood: maximum absolute logLik discrepancy **5.82e-11**.
- Max parameter differences by type from `compare_agreement_by_type.csv`: 1PL b 2.55e-6; 2PL a 8.82e-6, b 5.49e-6; 3PL a 8.26e-5, b 5.03e-5, c 1.71e-5; 4PL a 0.00409, b 0.01101, c 0.00428, d 0.000677.
- M25 lower asymptote: Python c=0 vs R c≈.0043 due to boundary parameterization. Python KKT per respondent < 1e-6; evaluation of R solution ~1.7e-4.
- 4PL truth-recovery RMSE: a 0.356 (Python), b 0.187, c 0.093, d 0.059. This is **one simulation dataset**, not a population-wide accuracy guarantee.

## Pilot Monte Carlo (5 replications per sample size)
| N | Python / R reported convergence | Mean Python boundary parameters | Mean 4PL a RMSE |
|---|---|---|---|
| 500 | 5/5 / 5/5 | 6.4 | 1.33 |
| 1,000 | 5/5 / 5/5 | 3.2 | 1.02 |
| 2,500 | 5/5 / 5/5 | 1.2 | 0.46 |

In one N=500 replication, R fitted M20 focal b=11.6 while Python enforced b≤4, and resulting logLik differed by ~0.086. Therefore **the implementations are not exactly constraint-matched** in every condition. Five simulations per N are inadequate for interval coverage or general sample-size cutoffs.

## Do not claim yet
- No independently verified Hessian standard errors, parameter confidence intervals, Wald/LR p-values, or 4PL boundary-aware inference.
- The supplied ZIP includes R/Python scripts and CSVs, but this CI did not execute them. The reference is **externally supplied evidence**, not automated regression fixtures.
- A fixed N threshold such as 1,000 cannot certify 4PL reliability; sample information, group balance, boundaries, effective item structure and curvature matter.
- The mixed engine is not connected to the Streamlit production UI.

## Next acceptance gates
1. Independently validate observed-information/Hessian SEs for interior parameters and explicitly withhold ordinary Wald inference when active boundaries/ill-conditioning invalidate it.
2. Include independent R estimates under documented matched-or-unmatched constraints, and rerun reproducibly with pinned versions.
3. At least 100 Monte Carlo repetitions per condition across multiple true-parameter designs; report convergence, RMSE, bias, boundary rates, SE coverage and failures.
4. Develop an opt-in experimental mixed-model UI with per-item type assignment, explicit anchors/DIF and diagnostics. Preserve legacy 1PL–4PL UI.
