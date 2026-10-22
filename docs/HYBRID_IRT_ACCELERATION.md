# Hybrid IRT accelerator — adoption gates

**Critical finding (2026-10-22):** PyPI [mirt 1.2.0](https://pypi.org/project/mirt/) by Cameron Lyons, MIT, is a **separate Python implementation** from R `mirt`. It advertises Rust-accelerated 1PL–4PL, GRM/PCM/GPCM/RSM/NRM, MIRT, mixed item types, bifactor, DIF, and other features. Feature claims are NOT equivalent to demonstrated scientific agreement.

## Fastest scientifically responsible plan
1. Preserve `core/` legacy, CTT, and all Streamlit pages. Do not replace old implementations.
2. Create an isolated Python 3.11 environment and pin `mirt==1.2.0`; test installation/wheel size/startup, importing `fit_mirt`, `simdata`, `fscores`, `get_backend_info`. Do **not** add to production `requirements.txt` before environment tests pass.
3. Run independent R `mirt` benchmarks on fixed, versioned test datasets: 1PL/2PL, GRM, PCM, GPCM, RSM, NRM; then 3PL/4PL, mixed, DIF, multigroup, MIRT. Compare same parameterization and identification, fixed-parameter logLik, free fit, score derivatives, convergence, boundary rates, SE, information and CI coverage. Summarize precise tolerances and exceptions per model.
4. Wrap external engines behind standardized interface (data schema, model, engine, parameters, latent metric, logLik/AIC/BIC, numerical diagnostics, missingness, seed, package version, reproducibility manifest). Never pass through unsupported SE/p-values.
5. Once independently validated, build one Streamlit dropdown-based UI. Use per-model availability/certification gates. Keep `py-irt` heavy Torch/Pyro backend optional, `girth` alternative only if its tests pass.
6. Keep Monte Carlo B12.7 independent-vs-testlet simulation ongoing as a separate task; preserve per-replication provenance.

## Restrictions
- No automatic fit-engine fallback after errors: user must know which engine ran.
- No commercial or deployment license claims unless versioned upstream licenses are recorded (Python mirt and girth MIT; py-irt MIT; R mirt GPL implications are separate).
- A successful unit test alone is not psychometric validation; high-quality model-wise benchmarks and uncertainty checks are compulsory.
- No possible design can guarantee outputs will never be questionable when data are uninformative. The software must explain and reject indefensible inferences.

## Immediate target
First integration is an **optional and gated** Python mirt adapter; if baseline checks fail, no release blocker for legacy EserPsychometrics. After scientific benchmark, promote model-by-model rather than one global production flag.
