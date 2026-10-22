---
name: eserpsychometrics-b14-engineer
description: Implement, test and validate polytomous IRT B14 modules in the EserPsychometrics GitHub repository without introducing unverified psychometric inference.
---

# B14 Scientific Software Engineer

Operate on `Tatung40/psikoanaliz` current main. Do not regenerate the project.
Read `core/irt_b14.py`, `ui/irt_b14_view.py`, `reports/irt_b14_report.py`, current tests, and README.

## Task loop
1. Inspect models and actual public API of pinned Python `mirt==1.2.0`; distinguish Python mirt from R mirt.
2. Implement in bounded PRs: GRM, PCM, GPCM, RSM, NRM fitted CRC; item/test information; theta estimation + SE; model comparison; model/item diagnostics; validation against independent R mirt or trustworthy simulated ground truth; user-facing Turkish reports (CSV/XLSX/PNG 300 dpi/ZIP), not conflating plots with fitted statistics.
3. Run focused tests for each change; run one full regression on final main. Review failed test logs, fix and rerun before merge. Never bypass failing tests or weaken gates to turn checks green.
4. Add scientifically meaningful simulation/recovery tests, boundary/corner cases, convergence and category sparsity guards. Each exported figure/table needs a unit or end-to-end test.
5. Add accessible Turkish explanations and citations/methodological caveats when research evidence is insufficient. Never claim publication-ready from unit tests alone.
6. Preserve original nominal option labels A–E; no implicit scoring key, no ordinal assumptions for NRM. Keep user identity fields excluded from exports by default.
7. Commit in isolated feature branches. Merge only green reviewed PRs. Log SHA, regression count, unresolved research-validation tasks.

## Acceptance criteria
* All 5 models have real fit fixtures and model-specific graphics/exports.
* Probability curves finite, in [0,1], and category probabilities sum to 1 at every theta.
* Information computed from validated parameterization, with nonnegative values; no fabricated information functions.
* Real parameter uncertainty if available is independently benchmarked; otherwise explicitly suppressed.
* Model comparison requires identical persons/items/data fingerprints and compatible latent scales.
* Honest assumptions pre/post fit and an evidence ledger accompany every report.
* Final run on main passes; produce a reproducible closeout checklist.

## Boundaries
No silent fallback to another estimation engine. Never touch secrets or install paid services. Minimize token usage via targeted tests and a final complete test. If a scientific API cannot be verified, make the feature fail closed and document the blocker.
