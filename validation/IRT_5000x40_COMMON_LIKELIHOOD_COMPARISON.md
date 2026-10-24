# 5000 x 40: common 61-node missing-aware likelihood comparison

Source: locally held synthetic 5000x40 matrix, 19,429 NaN entries, known generating 4PL parameters, and previously produced independent SciPy parameter CSVs (2PL, 3PL, 4PL). No individual response records are published.

All likelihoods recomputed with **the same** normalized N(0,1) 61-point Gauss-Hermite integration, scipy.special.logsumexp, and observed-response factors only. Parameterization: P(theta)=c+(d-c)*logistic(a(theta-b)).

| Parameter set | Marginal log likelihood |
|---|---:|
| Known generating 4PL parameters | -109756.570851 |
| Independent SciPy 2PL estimates | -109868.119100 |
| Independent SciPy 3PL estimates | -109748.823079 |
| Independent SciPy 4PL estimates | -109679.038234 |

This is **not** a comparison of Python mirt vs the native platform engine. The independent SciPy CSVs have not been certified for convergence. A fitted likelihood above the true generating parameters is possible due to sampling variability and parameter flexibility.

**4PL diagnostic warning:** maximum fitted a=11.596879, maximum fitted c=0.446338, minimum fitted d=0.508157. These violate the native engine's default constraints (a<=4, c<=.35, d>=.65), so this independent fit cannot be treated as an apples-to-apples native-engine comparison.

**Next mandatory gate:** compute a common-constraint, common-missingness fit with the Python mirt library and native engine and compare parameter recovery and log likelihood. AIC/BIC require consistent effective sample and parameter counts, and no claims of convergence should be made without optimizer diagnostics.
