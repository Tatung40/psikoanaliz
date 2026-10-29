# Independent R mirt — unidimensional GRM, PCM, GPCM, NRM

Gap closed: unidimensional polytomous Python mirt fits (B14 and "MTK: Python mirt" screens) had Monte Carlo
recovery evidence (`validation/POLYTOMOUS_MC_VALIDATION.md`) but no cross-engine comparison; R evidence existed
only for binary models (`validation/r_mirt_phase1`) and 2D–5D profiles (`validation/mirt_stage1`).

Design (fixed before any R run): example datasets level 0 (N = 300, 8 items, 4 categories); ML, no priors,
θ ~ N(0, 1). Python mirt 1.2.0: 61 Gauss–Hermite nodes, tol 1e-8. R mirt 1.48: 101 equally spaced points on
[-6, 6], TOL 1e-8. GRM ↔ `graded` (d = −a·b); PCM ↔ `gpcm` with a1 fixed 1; GPCM ↔ `gpcm` (d_k = −a Σ b_v);
NRM ↔ `nominal` (slopes a1·ak, ak0 = 0, ak3 = 3, d0 = 0).

Comparisons: R → Python mapping against R's own `probtrace` (1e-10); parameters; category probabilities on an
81-point θ grid; marginal log-likelihood of both parameter sets under one independent 121-node GH rule;
EAP and posterior SD; free-parameter count and convergence. Criteria: `validation/mirt_stage1/criteria.json`,
unchanged. R runs only in `.github/workflows/r-mirt-polytomous.yml`; without Rscript the comparison reports
NOT VERIFIED and fails. Python-side formulas and mapping: `tests/test_r_mirt_polytomous.py`.
