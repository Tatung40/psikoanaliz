# Common-boundary audit of independent SciPy IRT fits

Dataset: synthetic 5000 x 40, 9.7145% missing, response-level records not published. Recomputed on the actual locally held dataset with 61-point normalized Gauss-Hermite marginal likelihood.

| Independent fit | LL | a > 4 | c > .35 | d < .65 |
|---|---:|---:|---:|---:|
| 2PL | -109868.119100 | 0 | 0 | 0 |
| 3PL | -109748.823079 | 0 | 1 | 0 |
| 4PL | -109679.038234 | 1 | 2 | 2 |

These are counts of violations against **native default** bounds, not necessarily intrinsic model invalidity. In particular, Python mirt may use different defaults. A likelihood advantage under wider bounds is not a controlled engine comparison. The 4PL has at least one a estimate >4, two c estimates >.35, and two d estimates <.65.

Before deciding which engine to expose by default:
1. Verify both engines use equivalent IRT parameterization, latent distribution and missingness handling.
2. Refit under common bounds where supported, rather than merely clipping existing estimates.
3. Report convergence status, boundary counts, parameter recovery and independent marginal LL.
4. Distinguish predictive/model fit from recovery of the known simulation parameters.
5. Do not certify the library solely from successful smoke tests.
