# 5000 x 40 synthetic IRT: independently computed reference log likelihood

**Data:** locally held synthetic matrix, 5000 respondents x 40 items, 200,000 potential responses; 19,429 missing (9.7145%), 180,571 observed, with 84,608 incorrect and 95,963 correct observed responses. The known generating 4PL parameters span a=0.558..1.881, b=-1.913..2.322, c=0.002..0.280, d=0.788..0.995.

**Independent calculation:** for each respondent, the missing-response contributions are omitted from the conditional Bernoulli log likelihood, and the latent N(0,1) distribution is integrated using normalized Gauss-Hermite quadrature and scipy.special.logsumexp. Parameterization P(theta)=c+(d-c)*logistic(a*(theta-b)).

| Gauss-Hermite points | Marginal log likelihood evaluated at known true generating parameters |
|---:|---:|
| 21 | -109756.7621 |
| 31 | -109756.5503 |
| 41 | -109756.5676 |
| 61 | -109756.5709 |

**Numerical convergence:** 41 and 61 points differ by 0.0033 log-likelihood units over the entire dataset. This is an independent **likelihood reference at true parameters**, not a fitted model, not a recovery test and not a certification of Python mirt or the native EM. A fitted 4PL likelihood can exceed the likelihood at true parameters in a finite sample.

**Next verification:** Evaluate both Python mirt and native 4PL estimated parameter sets with the same independent likelihood routine, after ensuring the estimators use identical missing-data handling and parameterization. Check multi-start convergence, boundary counts and recovery. Never use complete-case vs missing-aware log likelihoods as if directly comparable.

The synthetic respondent-level matrix is intentionally not committed to this public repository.
