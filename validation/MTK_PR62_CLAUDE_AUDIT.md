# PR #62 — independent audit, fixes and evidence (Claude, 24 Ekim 2026)

Base: `68bcfda` (Codex delivery). No merge to `main`. Every number below is read from a committed
evidence file produced by a script whose design and acceptance criteria were fixed before running.

| Area | Report | Evidence |
|---|---|---|
| 3PL/4PL recovery root cause, MAP option, usage limits | `IRT_3PL4PL_ROOT_CAUSE.md` | `evidence/p1_3pl4pl/` (880 fits) |
| DIF: MH, Mantel, LR, ordinal LR, BH/BY/Holm/Bonferroni, IRT-LR coverage, 3PL/4PL bootstrap | `DIF_VALIDATION.md` | `evidence/p2_dif/` (2,560 analyses), `evidence/p2_dif34_bootstrap/` (1,300 replications) |
| Complete RSM chain | `RSM_VALIDATION.md` | `evidence/p3_rsm/` (1,020 fits) |
| GRM/PCM/GPCM/NRM Monte Carlo, NRM SE defect | `POLYTOMOUS_MC_VALIDATION.md` | `evidence/p4_polytomous/` (500 fits) |
| Real-browser E2E | this file | `evidence/p5_browser/`, `evidence/p5_browser_binary/` |
| Legacy engine reachability | this file | `evidence/legacy_reachability.json` |

## Browser verification

Playwright + Chromium against a real Streamlit server (`validation/browser_e2e_polytomous_dif.py`):
CSV upload through the file input, model selection, run, chart presence, real CSV/XLSX downloads,
reopening and comparison with a direct computation through the same engine path.

| Flow | Result | Max difference of reopened file vs direct computation |
|---|---|---|
| RSM (4 charts, XLSX 7 sheets, CSV) | PASS | δ 4.4e-16, τ 0 |
| Observed-score DIF MH (chart, CSV, XLSX) | PASS | p 5.6e-17; planted item flagged |
| Observed-score DIF LR | PASS | p 1.1e-16; planted item flagged |
| B14 GRM / PCM / GPCM / NRM (params CSV, persons CSV, XLSX) | PASS | parameters ≤ 8.9e-16, θ ≤ 4.4e-16 |
| 1PL–4PL (existing CDP smoke, MAP default for 3PL/4PL) | PASS | 8 items, finite a/b |

The real-browser run found a regression the unit tests had missed (policy columns broke 3PL/4PL curve
rendering); it is fixed and now covered by an AppTest.

## Legacy engine

Transitive import analysis from `app.py` and `pages/`: the retired native 1PL–4PL estimator
(`core.irt`, `ui.irt_optimization`, B10/B11 views) is reachable only through the B13 research page,
which now states this explicitly; a guard test fails if any other UI path reaches it. No module could
be deleted safely: every UI-unreachable module still has test/script consumers (26 modules). KTK and
Rasch code were not modified.

## Open scientific blockers (not hidden)

1. 3PL/4PL: parameter interpretation supported only with MAP at N ≥ 3000 (3PL ML N ≥ 10000; 4PL ML
   never), k ≥ 12, normal θ. SE/CI coverage of 3PL/4PL item parameters not validated.
2. 3PL bootstrap DIF size .076 for an interior studied item (criterion ≤ .075 missed) → p withheld;
   4PL warp-speed evidence covers only the studied design (k = 8, 500 + 500), not input-specific eligibility; both models’ inferential p-values are now withheld in UI/CSV.
3. Observed-score ordinal DIF under group impact: family-wise error .09–.12 (warned in UI).
4. Selection-adjusted inference after data-driven anchor purification; broader FDR/coverage designs.
5. NRM weighted category-effect intervals in `core.polytomous_dif` (previous 88 % coverage) remain
   withheld; Python mirt item-fit statistics for polytomous models not validated.
6. Independent R mirt comparison and the user's own real-data run are outside this task's scope.
