# PR #152 — Independent certification audit (2026-11-06)

Base: d7ec24b. Scope: source review of core/rater_reliability.py, core/rater_methods.json, PROTOCOL.json, compare.py, EVIDENCE.md, and UI export code. **This review did not execute Python/R or CI.** Existing 154 comparisons are reported by PR, not independently rerun in this audit.

## Verdict: NOT CERTIFIED / merge gate open

1. **P0 — Evidence leakage (fixed on audit branch).** `run()` previously allowed estimate, chi2, bias, F, df, loa and other outputs through irrespective of `validated_outputs`. UI's detailed CSV/XLSX/ZIP export independently serialized `out["result"]`, bypassing the gate entirely. Both now use the validated `shown` mapping; regression tests added.
2. **P0 — Validation cannot certify general data conditions.** `compare.py` evaluates one deterministic example per method and uses the same examples as the public teaching fixtures. Independent R implementations reduce implementation risk but do not establish coverage across degenerate, unbalanced, high-prevalence, tied, near-boundary and varying sample-size regimes. Do not relabel NUMERICALLY VALIDATED as CERTIFIED.
3. **P1 — ICC(2,k) CI reference disagreement.** Existing report says irr ICC(A,k) lower 0.910753 versus psych's Spearman–Brown transform 0.911490. The Python calculation agrees with psych only. Keep CI hidden until formula provenance, model estimand, degrees of freedom and multiple independent numerical fixtures resolve it. Never choose a reference solely because it matches Python.
4. **P1 — Evidence generation is not exhaustive.** `compare.py` marks a method validated if all *present* reference rows pass. It does not require every preregistered output to have a comparison or enforce minimum independent references per output. Missing rows can silently produce partial evidence. Add coverage assertions for all declared method/output combinations and explicitly label unsupported outputs.
5. **P1 — Input constraints.** `from_counts` lacks an explicit check that category columns exist and are unique; malformed schemas may raise a raw KeyError rather than a Turkish RaterDataError. `from_long` casts missing IDs to strings, potentially conflating missing IDs with the literal "nan". Validate both before certification.
6. **P1 — Statistical inference and applicability.** For weighted kappas, confirm that z0, SE and CIs use clearly documented null/alternative variance conventions. Kendall W p-value uses a chi-square approximation; specify conditions and tie treatment. Bland–Altman LoA intervals assume independent paired differences and approximate normality; explicitly document assumptions. Pearson/Spearman are association, not agreement; Stuart–Maxwell/Bhapkar/McNemar are marginal-homogeneity tests, not reliability coefficients.
7. **P1 — License and reproducibility.** Review GPL-3.0 obligations for the intended deployment/distribution, not merely whether another GPL dependency is present. Freeze psych and all transitive R versions and record package digests. Capture full environment provenance.

## Acceptance gates

- For each method × output, preregistered estimand, reference implementation, tolerance, supported data conditions, and explicit PASS/FAIL/NOT CERTIFIED.
- Deterministic boundary matrix: small n, moderate n, skewed prevalence, perfect/zero agreement, constant raters, category gaps, missingness, ties, singular tables, label permutations and numeric scale transformations where mathematically appropriate.
- ICC(A,k) interval adjudicated against published formula with two independent R/Python implementations and >1 nondegenerate dataset.
- CI tests green on branch; exported CSV, XLSX, DOCX, ZIP and UI all hide unvalidated results. Verify no raw result leaks in narrative/plot.
- No Monte Carlo required for deterministic formula validation. If coverage properties or interval coverage are claimed, separately preregister simulation design and seed.

**Status: audit fixes committed; CI execution and broader numerical validation not yet performed. No CERTIFIED labels authorized.**
