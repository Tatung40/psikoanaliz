# Python mirt 1PL–4PL: Scientific validation checklist

## Engine and scope
- Active binary calibration: Python `mirt==1.2.0`; the retired native binary engine must not run.
- Keep Rasch and other legacy research pages separate until their dependencies and diagnostics are audited.
- No unsupported native optimization sliders, silent imputation, or cross-engine equivalence claims.

## Independent numerical evidence
- `validation/known_truth_irt_recovery.py`: independent NumPy logistic generator with known item parameters; two fixed seeds; per-model and per-parameter bias, MAE, RMSE, max error, and convergence status.
- `validation/python_mirt_parameter_recovery.py`: library-generated simulation stability only; **not** true parameter recovery.
- `tests/test_irt_python_mirt_parameters.py`: direct coefficient extraction and independent logistic/finite-difference curve checks.
- `validation/inspect_python_mirt_api.py`: records actual coefficient schema.
- `validation/audit_legacy_irt_imports.py`: inventories remaining dependencies; **not** a license to delete modules.

## UI and reproducibility
- `tests/test_irt_binary_apptest.py`: executable Streamlit AppTest for 1PL–4PL controls and native-button absence. This is **not** browser automation.
- Left sidebar: model, theta grid, item selector, report toggles and execution.
- Right results: parameter CSV/XLSX, diagnostics CSV, per-item ICC/IIF, TIF, conditional SEM and bulk curves CSV.
- Validate downloads by opening the resulting files and checking dimensions, finite values and labels.

## Release blockers
1. Review CI artifacts for known-truth recovery: report bias/MAE/RMSE and predeclare acceptable thresholds.
2. Compare coefficients and fit indices with an independent engine (for example R mirt), using identical data and documented parameterization.
3. Run a genuine browser E2E test covering upload, selection, fitting, plots and downloads.
4. Audit/remove remaining legacy native modules only after consumers are migrated; keep Rasch/KTK intact.
5. Require all scientific and general CI workflows to pass before merging PR #62 into main.

No scientific certification or complete end-to-end validation is asserted by this checklist.
