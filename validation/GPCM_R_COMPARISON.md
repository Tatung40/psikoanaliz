# GPCM independent validation profile

Reuses PR65/69's common driver and no-fit preflight; PR71's shared person-table API is reused and hardened. Original branches/main are unchanged. Actual execution and artifact links are recorded in this PR, rather than certifying this design document as a fitted result.

- Existing 500×8 ordinal fixture, seed20261024, categories0..3; no new simulation. Positive per-item free a, unordered steps b, fixed latent N(0,1), D1, ML/no priors, no acceleration, explicit identical GH61 nodes/normal weights.
- Python mirt1.2.0 `GPCM`; a∈[.1,5], steps∈[-6,6]. R mirt1.48 `gpcm`, free a1 and d1..d3, fixed d0=0 and ak0..ak3=0..3. This is distinct from PCM's fixed a=1.
- R native d_s=−a Σ_(t≤s)b_t; b_s=−(d_s−d_(s−1))/a. Comparing raw intercepts to steps is prohibited. Forward/inverse Jacobians, probabilities, information, exact affine step constraints and slope endpoints are checked before fits.
- Full R constraints enforce −6a≤d_s−d_(s−1)≤6a. Native |d|≤90 is a redundant outer box, not a substitute for physical step bounds. Coordinate CSV types are explicitly restored as in PCM.
- One fit per engine/attempt, no retries. Actual nodes, prior, identification, convergence, native/common-grid logLik, AIC/BIC, physical/native gradients and boundary flags are recorded. Parameter differences include every item/a/step with absolute and relative discrepancies. No scientific cutoff is invented.
- Independent R EAP/posterior SD uses **fitted R mirt probtrace** and the explicitly shared GH61 normal weights; it does not use R fscores's potentially different default quadrature. Python uses the shared PR71 API with those weights. The R and Python programs calculate their own posterior means/variances. The R-trace replay checks scoring algebra separately from calibration differences, with the existing sqrt(machine epsilon) algebra-only criterion.
- Person identifiers are generated row IDs for the public fixture. In production, real supplied IDs are preserved positionally, including duplicate dataframe indices; duplicate/missing IDs are rejected. Explicit item and category mappings prevent curve sorting from reassigning responses. True zero probabilities remain zero; impossible patterns fail. Person outputs require converged fits.
- Production GPCM person tables use GH61 (no truncated [-4,4] posterior); other existing numerical scoring defaults remain unchanged. CSV/Excel and the Streamlit table share the same unrounded source frame. Conditional SEM, person posterior SD and parameter SE are distinct.

Reproducible command after successful actual no-fit preflight:

```sh
python validation/poly_r_preflight.py --output preflight-output
python validation/grm_r_comparison.py --model GPCM --preflight preflight-output --output preflight-output/gpcm-fit --execute
```

The existing Actions workflow has `gpcm-fit`; this branch's PR selects it automatically. All original models' browser checks remain enabled. Logs, raw estimates, category/IIF/TIF traces, EAP/SD/person comparisons, software versions and hashes upload even on failure.

A model's convergence flag is not a scientific-equivalence certificate. Existing cross-engine acceptance criteria have not been independently calibrated. Quantitative agreement must not be labelled scientific PASS merely because pytest/CI passes. This profile does not establish population-wide recovery or SE/CI coverage. Earlier successful models are not independently recalibrated by this workflow.
