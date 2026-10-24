# 3PL/4PL parameter recovery: root cause and usage limits

Script: `validation/irt_3pl4pl_root_cause.py` (design and acceptance criteria are in its docstring and
were fixed before any result). Evidence: `evidence/p1_3pl4pl/irt_3pl4pl_root_cause.json`
(880 fits, all 880 converged, 0 failed; Python mirt 1.2.0; independent NumPy generator).

Truth as in PR #62 (`known_truth_irt_recovery.py`), extended to k = 30; theta ~ N(0, 1) unless noted;
20 replications per cell. Pre-registered criteria: RMSE a/b <= .30, c/d <= .07, IRF RMSE <= .02,
<= 5 % parameters at a library box bound.

## What is NOT the cause (tested)

| Candidate | Test | Result |
|---|---|---|
| Parameter mapping / slope transformation | IRF from reported a,b,c,d vs `model.probability` | max diff 3.3e-16 |
| Quadrature / tolerance | ML 21 nodes tol 1e-4 vs 61 nodes tol 1e-6 | same estimates (a RMSE 1.82 vs 1.86, 4PL N1000 k12) |
| Local maxima / start values | ML started at the TRUE parameters | converges to the same maximum (a RMSE 1.87) |
| Optimizer failure | logL(estimate) − logL(truth), common 61-node grid | positive in every one of 880 fits (min +5.9) |

## Cause

Weak identification of a, c and d: the marginal likelihood is nearly flat along ridges where a larger
slope is compensated by a lower upper / higher lower asymptote. Every ML fit placed parameters on the
library box (a = 5, c = 0, d = 1); estimation errors of a and c are correlated (mean r = .56, 4PL
N1000 k12). Item response functions are nonetheless recovered (IRF RMSE .016–.051), so different
(a, b, c, d) give practically the same curve. More items and persons shrink but do not remove it for
ML 4PL. A non-normal latent distribution (standardized chi-square(4)) adds large bias.

## Before / after (mean over 20 replications; theta normal)

| Cell | ML (PR #62 setting) a / b / c / d RMSE | MAP library default priors a / b / c / d RMSE | Bound fits ML → MAP |
|---|---|---|---|
| 3PL N1000 k12 | .399 / .377 / .128 / — | .211 / .151 / .049 / — | 20/20 → 1/20 |
| 3PL N3000 k30 | .158 / .222 / .079 / — | .129 / .137 / .047 / — | 20/20 → 0/20 |
| 3PL N10000 k30 | .086 / .145 / .053 / — | .078 / .100 / .036 / — | 19/20 → 0/20 |
| 4PL N1000 k12 | 1.816 / .562 / .164 / .122 | .242 / .206 / .047 / .040 | 20/20 → 0/20 |
| 4PL N3000 k30 | .462 / .294 / .099 / .064 | .166 / .170 / .042 / .035 | 20/20 → 0/20 |
| 4PL N10000 k30 | .240 / .202 / .078 / .045 | .140 / .126 / .041 / .029 | 20/20 → 0/20 |
| 3PL N3000 k30, skewed theta | .641 / .698 / .221 / — | .443 / .560 / .180 / — | — |
| 4PL N3000 k30, skewed theta | .819 / .752 / .196 / .055 | .414 / .530 / .136 / .027 | — |

The PR #62 figure (4PL a RMSE 2.39 from two seeds) is of the same order as the 20-replication ML value
(1.82); two seeds were too few to estimate it.

## Pre-registered criteria met

| | N = 1000 | N = 3000 | N = 10000 |
|---|---|---|---|
| 3PL ML | no | no | yes (k 12, 30) |
| 3PL MAP | no (IRF RMSE .023–.025) | yes (k 12, 30) | yes |
| 4PL ML | no | no | no |
| 4PL MAP | no (IRF RMSE .026–.028) | yes (k 12, 30) | yes |

## Applied changes

* `fit_python_mirt(..., estimation="MAP")` for 3PL/4PL uses `mirt.estimation.priors.default_priors`
  (library defaults, not tuned). MAP is the default on the 3PL/4PL pages; ML stays selectable.
* `core/irt_binary_usage_policy.py` encodes the table above; the 3PL/4PL pages show
  supported / caution / not supported, and parameter exports carry the estimator, the policy status
  and the parameters that sit on a library bound.
* MAP estimates are Bayes modal: logLik/AIC/BIC are not prior-free ML criteria and prior means pull
  weakly informed parameters (c prior mean .2). This is stated in the UI.

## Limits

Normal latent distribution assumed; N < 1000, k < 12 and non-normal theta are outside the supported
region. SE/CI coverage of 3PL/4PL parameters was not part of this study and remains unvalidated.
