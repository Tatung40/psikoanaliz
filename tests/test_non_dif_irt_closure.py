"""Frozen closure inventory: zero calibrations, no statistical gate changes."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_non_dif_complete_matrix_counts_and_sources():
    m=json.loads((ROOT/"validation/NON_DIF_IRT_CLOSURE.json").read_text())
    counts={k:0 for k in ("CERTIFIED","PARTIAL","NOT CERTIFIED")}
    assert len(m["rows"])==34 and len(m["columns"])==10
    for row in m["rows"]:
        assert "DIF" not in row["model"].upper()
        assert set(row["cells"])==set(m["columns"]) and row["scope"] and row["notes"]
        for value in row["cells"].values():counts[value]+=1
        for path in row["evidence"]:assert (ROOT/path).is_file(),path
        if row["model"].startswith("MIRT"):
            for key in ("Parameter SE","Parameter CI","Person uncertainty","Inferential fit","Model comparison"):
                assert row["cells"][key]=="NOT CERTIFIED"
    assert counts==m["cell_counts"]==m["cell_counts_after"]==m["cell_counts_before"]
    assert counts=={"CERTIFIED":87,"PARTIAL":161,"NOT CERTIFIED":92}
    assert m["scoped_inference_groups"]==dict(before=29,after=31,P1=14,P2=6,P3=2,P4_C1=7,new_global=2)
    assert not m["full_certification"] and m["no_new_numerical_campaign"]
def test_original_certificates_adapter_q1_manifest_and_dif_unchanged():
    lock=json.loads((ROOT/"validation/NON_DIF_IRT_PROTECTED_SOURCES.json").read_text())
    for item in lock["sources"]:
        raw=(ROOT/item["path"]).read_bytes()
        assert hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()==item["git_blob"],item["path"]
def test_original_seven_p4_gates_and_two_archived_global_references():
    from core.polytomous_certification import BLOCKS,_confirmed
    assert sum(map(len,BLOCKS.values()))==7
    for model,blocks in BLOCKS.items():
        for block in blocks:assert _confirmed(model,block),(model,block)
    proof=json.loads((ROOT/"validation/evidence/binary_global_fit/CERTIFICATION.json").read_text())
    assert proof["canonical_run"]==45856527113
    for model in ("3PL","4PL"):
        p=proof["models"][model]
        assert p["null_R"]==6000 and all(v is True for v in p["gates"].values())
        assert p["refusal"]==0 and p["status"]=="SCOPED CERTIFIED"
        assert "nonmonotone" not in p["certified_claims"]
    assert "second_factor" not in proof["models"]["4PL"]["certified_claims"]
