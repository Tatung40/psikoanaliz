# Dichotomous Rasch module — validation record (B8)

**Scope:** dichotomous Rasch model only (`core/rasch.py`, Streamlit page "📏 Rasch Analizi").
PCM, RSM and MFRM are out of scope (see "Future stages").

## Method

| Element | core.rasch | R mirt 1.48 (reference) | RaschPy 1.4.0 SLM (second reference) |
|---|---|---|---|
| Item estimation | MML, direct L-BFGS-B with analytic score | MML/EM | **PAIR** (Choppin pairwise, Bradley–Terry) — not MML |
| Scale origin | latent mean 0, latent SD free | same (run as 2PL with equal slopes, N(0,1); a = σ, d = −b) | items zero-centred |
| Quadrature | Gauss–Hermite, 41 nodes (default) | identical GH-41 grid (`customTheta`/`customPriorFun`) | none |
| Persons | Warm WLE (finite for extreme scores) | `fscores(method='WLE')` | Warm WLE |
| Missing data | skipped in all likelihoods (MAR) | skipped (NA) | skipped |
| Item SE | observed information of the marginal likelihood | Hessian of mirt's logLik (`numerical_deriv`) | sandwich (PAIR estimating equations) |
| Fit | Wright & Masters Infit/Outfit, Wilson–Hilferty ZSTD, extreme persons excluded from item fit | same formulas with conventions below | same formulas, extremes excluded |
| Reliability | separation reliability with model SE (persons: WLE, extremes excluded); marginal EAP reliability | – | PSI with residual-inflated RSEM |

Convergence is decided from the score norm (≤ 1e-6 per person), not from the optimizer
flag. SEs are refused if the estimate did not converge or the information is not
positive definite. Flags: MNSQ outside the chosen range (default 0.7–1.3, Wright &
Linacre, 1994) **and** |ZSTD| > 2 → "İnceleme önerilir" (a review suggestion, not a verdict).

Fixtures (`scripts/make_rasch_fixtures.py`): `rasch_main` k=10, N=1000, σ=1.2, items 3/8
generated with slopes 0.4/2.5 (misfit signal); `rasch_missing` k=10, N=800, pure Rasch,
11.9% MCAR missing. Reproduce: `Rscript scripts/mirt_rasch_reference.R`,
`python scripts/raschpy_reference.py`. Tests: `tests/test_rasch_core.py`,
`tests/test_rasch_external_reference.py`, `tests/test_rasch_streamlit.py`.

## Results

| Check | Result | Class |
|---|---|---|
| logLik at mirt estimates (both fixtures) | rel. diff ≤ 1e-12 | PASS |
| Free MML vs mirt: b, σ, logL | max \|Δb\| 1.2e-8, \|Δσ\| 3e-9, ΔlogL 4e-12 | PASS |
| SE(b), SE(σ) vs mirt logLik Hessian (Jacobian incl. 2nd-order term) | rel. diff ≤ 2.7e-9 | PASS |
| WLE vs mirt `fscores` | ≤ 3.8e-6 (mirt's root tolerance; ours equals a direct root to 1e-10) | PASS |
| Infit/Outfit MNSQ vs mirt | non-extreme persons ≤ 1e-13; items ≤ 3e-5 (mirt floors W for extreme θ) | PASS |
| ZSTD vs mirt | differs; fully explained by mirt conventions: q capped at √2, W floored when W² < 1e-5, person q.outfit mixes J and observed item counts. Emulating these reproduces mirt to ~1e-13 | WARNING (convention) |
| mirt item Outfit ZSTD with missing data | R vector-recycling defect in `itemfit` (`(C/W^2)/N^2` with per-item N); emulated exactly; complete data unaffected | FAIL in mirt 1.48 (not in core.rasch) |
| mirt `itemtype='Rasch'` on its default 61-node grid (information) | \|Δb\| ≤ 1.3e-5, \|Δσ\| ≤ 2.8e-5 (quadrature difference) | PASS (info) |
| WLE given RaschPy items | ≤ 7e-12 | PASS |
| Infit/Outfit given RaschPy estimates | persons all four ≤ 4e-11; items MNSQ and Infit ZSTD ≤ 3e-11 | PASS |
| RaschPy item Outfit ZSTD | uses n incl. extreme persons in q while MNSQ excludes them; reproduced only under that convention (≤ 2e-11); ours keeps one consistent n | WARNING (RaschPy inconsistency) |
| PAIR (RaschPy) vs MML item locations, centred | r ≥ 0.9987, max \|Δ\| 0.195 (main) / 0.044 (missing) — different estimators, not equivalent | reported only |
| Person WLE with own items: ours vs RaschPy | r ≥ 0.99999 | reported only |
| Reliability vs RaschPy | persons 0.584 vs 0.705 (main), 0.486 vs 0.626 (missing); items 0.995 vs 0.997 — different SE/RSEM and extreme-person conventions | WARNING (definition) |
| Misfit detection | slope-2.5 item: Outfit 0.46, ZSTD −7.1; slope-0.4 item: Outfit 1.31, ZSTD 5.0 → both flagged at 0.7–1.3 | PASS |
| RaschPy documentation | docstrings name `diffs`, `person_locations`; attributes actually set are `items`, `persons` | noted |

## Open items and future stages

- Not tested: very short tests (k < 10) for ZSTD stability, item-level DIF, local
  dependence (Q3), anchoring/linking, other quadrature sizes against mirt.
- Person reliability with residual-inflated ("real") SE is not offered.
- Future stages (not in B8): PCM, RSM, MFRM.
