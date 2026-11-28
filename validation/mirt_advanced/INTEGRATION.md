# Advanced MIRT integration contract (applied after the Codex core MIRT PR merges)

No Codex-owned file has been changed. The shared integration still to do:

1. **UI/status (Codex-owned `ui/irt_multidim_view.py`, `core/irt_multidim.py` MODELS):**
   - Add "GPCM bifactor" and "GRM bifactor (high precision)". Both call `core.mirt_advanced.fit_polytomous_bifactor` with groups; the GRM entry uses `high_precision=True`.
   - Label both EXPERIMENTAL / NOT CERTIFIED.
   - PCM bifactor stays hidden (WITHHELD).
2. **GRM high-precision profile:** an explicit user option that calls `core.mirt_advanced.fit_grm_high_precision`. It must not run automatically.
3. **Native R engine option for fixed M3PL/M4PL:**
   - Select it explicitly. When `engine_available()` is False, show the refusal message; never fall back silently.
   - Report the `engine`, `mirt_version` and `validation_scope` metadata. Do not show item SE, CI or p-values.
4. **Exports:**
   - Export the parameter `values`, `covariance`, `EAP` and `posterior_SD`, labelled as person-score uncertainty and not item SE.
   - Export the engine metadata and the status label.
5. **`mirt-stage1` workflow repair (Codex-owned):** the step "Reuse retained converged fits" downloads expired artifact 40499970808.
   - Smallest patch: replace that download and `--reuse` step with one fresh `compare.py --output … --expand` run. It regenerates the same seeded fixtures; `data_sha256` is checked against `criteria.json`.
   - Then upload the result with a long retention period.
   - Historical evidence JSON stays unchanged.
6. **Deployment:** the Docker image installs pinned R mirt 1.48 at build time. Streamlit Community Cloud does not build Dockerfiles. Making native R available to live users needs an owner-authorised move to a Docker host; no paid service is introduced here.

## Status after Codex #211 merged (main cb84fc5)
- Items 1–4 are integrated in `ui/irt_multidim_view.py` through `core/mirt_advanced_report.py`, on the shared `{metadata, tables}` contract.
  - Every route reuses `core.irt_multidim.report` or the general SciPy curve tables, and `reports.irt_multidim_report.downloads`.
  - Advanced routes store `outcome=None`, so Codex's numerical-scope decorator never relabels them.
  - Item SE/CI/p and fit inference are `WITHHELD` in `Output_scope`.
  - Polytomous bifactor is restricted to G + exactly 3 specific factors.
- Item 5 (stage-1 workflow): Codex #211 audited the lost artifact (`validation/mirt_core_production/REPORT.md`, "Lost archive") and keeps durable evidence in the repository. No further shared patch is applied here.
- Item 6: unchanged. Docker is verified; Community Cloud has no native R.
- Tests: `tests/test_mirt_advanced_integration.py`, using actual AppTest UI and CSV/XLSX/PNG/ZIP artifacts. The CI `integration-r` job runs it with R mirt 1.48 present.
