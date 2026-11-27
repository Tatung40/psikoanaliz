"""Frozen evidence/source integrity and portable policy inputs; no fits."""
import io,json,zipfile
from pathlib import Path
from core import reproducible as R
from validation.mirt_core_production.audit import audit
def test_all_historical_evidence_and_owned_model_sources_preserved():
    report=audit()
    assert report["new_calibrations"]==report["new_R_calibrations"]==0
    assert len(report["profiles"])==7 and report["inference"].startswith("WITHHELD")
def test_mirt_reproducibility_bundle_contains_scope_evidence_and_policy():
    calls=[dict(module="core.irt_mirt_production",function="decorate_native")]
    files={str(p.relative_to(R.ROOT)) for p in R._runtime_assets(calls)}
    expected={"validation/mirt_stage1/criteria.json","validation/mirt_stage1/phase1_evidence.json",
              "validation/mirt_stage1/expansion_original_evidence.json","validation/SCIENTIFIC_CERTIFICATION_POLICY.md"}
    assert expected<=files
    hashes=R.runtime_hashes(calls)
    assert expected<=set(hashes)
    manifest={"packages":{},"calls":calls,"runtime_sha256":hashes}
    with zipfile.ZipFile(io.BytesIO(R.bundle(manifest,"# no model fit in this portability test"))) as archive:
        for path in expected:
            assert archive.read("runtime/"+path)==(R.ROOT/path).read_bytes()
def test_derived_status_and_score_moments_do_not_count_as_fresh_calibrations():
    from core.academic_screen_results import primary
    for module in ("core.irt_mirt_policy","core.irt_mirt_production","core.irt_mirt_diagnostics"):
        assert not primary(dict(module=module,function="refresh_policy"))
    assert primary(dict(module="core.irt_multidim",function="fit"))
