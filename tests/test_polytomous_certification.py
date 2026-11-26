from core.polytomous_certification import ci_status
S = dict(max_iter=500, tol=1e-4, n_quadpts=21)


def test_scope_enforced():
    assert ci_status("GRM", "discrimination", 2000, "ML", S, n_items=10, n_categories=4).startswith("SCOPED CERTIFIED")
    assert ci_status("NRM", "slopes", 2000, "ML", S, "oakes", 10, 4).startswith("SCOPED CERTIFIED")
    for args in [("GRM", "discrimination", 500, "ML", S), ("RSM", "steps", 2000, "ML", S), ("NRM", "slopes", 2000, "ML", S, "auto"),
                 ("GPCM", "steps", 2000, "MAP", S), ("PCM", "steps", 2000, "ML", dict(S, n_quadpts=61)), ("GRM", "thetas", 2000, "ML", S)]:
        assert ci_status(*args) == "NOT CERTIFIED"


def test_unknown_metadata_fails_closed():
    assert ci_status("GRM", "discrimination", 2000, "ML", S) == "NOT CERTIFIED"
    assert ci_status("NRM", "slopes", 2000, "ML", S, None, 10, 4) == "NOT CERTIFIED"
    assert ci_status("GRM", "discrimination", 2000, "ML", S, n_items=12, n_categories=4) == "NOT CERTIFIED"
