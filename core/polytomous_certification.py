"""Scope-aware status for P4-C1 polytomous parameter-CI certificates (validation/evidence/p4_confirm/REPORT.json).
CERTIFIED only for the exact confirmed scope: models GRM/PCM/GPCM/NRM, N=2000, application fit path with defaults
(estimation ML, max_iter 500, tol 1e-4, n_quadpts 21, NRM SE = Oakes). Everything else (N500, RSM, other settings,
NRM non-Oakes/auto SE) -> NOT CERTIFIED."""
BLOCKS = {"GRM": {"discrimination", "thresholds"}, "PCM": {"steps"}, "GPCM": {"discrimination", "steps"}, "NRM": {"slopes", "intercepts"}}
SETTINGS = dict(max_iter=500, tol=1e-4, n_quadpts=21)


def ci_status(model, block, n_persons, estimation="ML", solver_settings=None, se_method=None, n_items=None, n_categories=None):
    """Unknown metadata (None) fails closed. Registered design: k=10 items, C=4 categories (polytomous_mc_study)."""
    ok = (n_items == 10 and n_categories == 4 and model in BLOCKS and block in BLOCKS[model] and n_persons == 2000 and estimation == "ML"
          and dict(solver_settings or {}) == SETTINGS and (model != "NRM" or se_method == "oakes"))
    return ("SCOPED CERTIFIED (P4-C1: N=2000, kayıtlı üreteç, varsayılan ML ayarları)" if ok else "NOT CERTIFIED")
