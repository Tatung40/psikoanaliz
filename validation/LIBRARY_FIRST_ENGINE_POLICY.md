# Library-first scientific engine policy

- Default preference: independently validated maintained Python library.
- Native estimators remain optional research/reference engines, not automatic defaults.
- Each model and missing-data condition requires its **own** certification.
- Package installation, passing unit tests and mathematical ICC/IIF agreement are not sufficient to certify MML parameter recovery.
- Certification gates: reproducible simulated-data recovery, convergence, likelihood agreement, boundary diagnostics, sensitivity to starting values, performance, missingness handling, model-specific tests.
- Never silently impute missing responses or treat them as zeros.
- If no engine is certified, block inferential reporting with a clear reason.
- `core/irt_engine_policy.py` is a policy layer only. **It is not yet connected to the Streamlit IRT execution route**, and no model is pre-certified.
- Existing platform results must not be represented as library-backed until routing and integration tests demonstrate it.
