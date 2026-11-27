from core.polytomous_certification import ci_status
S = dict(max_iter=500, tol=1e-4, n_quadpts=21)


def test_scope_enforced():
    assert ci_status("GRM", "discrimination", 2000, "ML", S, "oakes", n_items=10, n_categories=4).startswith("SCOPED CERTIFIED")
    assert ci_status("NRM", "slopes", 2000, "ML", S, "oakes", 10, 4).startswith("SCOPED CERTIFIED")
    for args in [("GRM", "discrimination", 500, "ML", S), ("RSM", "steps", 2000, "ML", S), ("NRM", "slopes", 2000, "ML", S, "auto"),
                 ("GPCM", "steps", 2000, "MAP", S), ("PCM", "steps", 2000, "ML", dict(S, n_quadpts=61)), ("GRM", "thetas", 2000, "ML", S)]:
        assert ci_status(*args) == "NOT CERTIFIED"


def test_unknown_metadata_fails_closed():
    assert ci_status("GRM", "discrimination", 2000, "ML", S) == "NOT CERTIFIED"
    assert ci_status("NRM", "slopes", 2000, "ML", S, None, 10, 4) == "NOT CERTIFIED"
    assert ci_status("GRM", "discrimination", 2000, "ML", S, n_items=12, n_categories=4) == "NOT CERTIFIED"

def test_missing_method_and_missing_or_withdrawn_evidence_fail_closed(tmp_path,monkeypatch):
    import json
    import core.polytomous_certification as C
    assert C.ci_status("NRM","slopes",2000,"ML",S,None,10,4)=="NOT CERTIFIED"
    original=json.loads(C.REPORT.read_text())
    target=tmp_path/"report.json";target.write_text(json.dumps(original));monkeypatch.setattr(C,"REPORT",target)
    before=C._evidence_fingerprint()
    assert C.ci_status("GRM","discrimination",2000,"ML",S,"oakes",10,4).startswith("SCOPED CERTIFIED")
    original["GRM discrimination"]["coverage"]=.90
    target.write_text(json.dumps(original))
    assert C._evidence_fingerprint()!=before
    assert C.ci_status("GRM","discrimination",2000,"ML",S,"oakes",10,4)=="NOT CERTIFIED"
    target.unlink()
    assert C.ci_status("NRM","slopes",2000,"ML",S,"oakes",10,4)=="NOT CERTIFIED"
