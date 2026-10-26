# NRM independent reference profile

The numerical budgets in `nrm_acceptance.json` are committed **before the first NRM R/Python calibration**. They specify reporting/computational accuracy, independently of the new fitted discrepancies. They are not externally calibrated clinical equivalence limits; a passing profile is MEASURED / NOT CERTIFIED, never population-wide scientific certification. Failed budgets remain failures and are not enlarged.

The existing PR72 driver, fixture, GH61 setup, artifact/reporting and PR71 person API are reused. Branches71/72/73 and main are not modified. R mirt1.48 and Python mirt1.2.0 remain independent engines. No new psychometric estimator is implemented.

Python NRM uses baseline-category A0=D0=0 and P_c=softmax(A_c*theta+D_c); free A∈[-5,5],D∈[-6,6], normal mean0/variance1,D1. R `nominal` is configured via its official values table: a1=1 fixed, ak0=0/d0=0 fixed, ak1..ak3 and d1..d3 free. Thus A=ak,D=d directly, including zero last-category slopes; no singular division by the default R score anchor is needed. Runtime checks must verify that R accepts/preserves this chart before any comparison is declared valid.

Categories retain explicit nominal labels; no thresholds, ordinal raw scores, Infit/Outfit/S-X² or Wright Map are inferred. Global theta reflection is a latent-orientation convention, not achievement direction. Each engine's reported orientation is chosen by its slope dot product with the same predeclared starts; near-zero orientation contrast blocks rather than forcing alignment. Raw coefficients are preserved. Curves/person theta are reflected consistently; likelihood and posterior SD are unchanged under the symmetric prior.

One fit per engine per attempt, same existing500×8 data/coding/starts/GH61/prior, ML/no item priors/no acceleration. Actual constraints, native/common logLik, IC identities, scores and convergence are retained. R person moments are evaluated independently from its fitted probtrace with identical GH61 normal weights, as in PR72; default fscores quadrature is not silently substituted. No ordinal score sum is used for persons.

Actual execution, per-category coefficients, probability/IIF/TIF/SEM/person comparisons, versions, hashes, tests and CI outcomes are recorded in the PR and its original artifacts. Same-parameter algebraic checks are distinct from separately fitted estimate agreement. If exact public-API identification/constraints cannot be established, classify BLOCKED and stop forcing the comparison.

NRM application and Python validation person scoring uses the supported `mirt.fscores(..., method="EAP", n_quadpts=61)` API. Its implementation uses the same normal GH quadrature class as calibration. Independent R posterior moments and a same-probability replay check test integration identities; they do not replace the Python library estimator. Posterior SD is distinguished from information SEM.
