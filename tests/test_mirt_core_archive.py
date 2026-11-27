"""Actual frozen cold-NRM archive: no fits or regenerated reference evidence."""
import ast
import hashlib
import json
from pathlib import Path
import pytest
from validation.mirt_core_production import archive as A


def test_actual_offline_archive_and_independent_native_coefficients():
    result = A.verify()
    assert result["archive_integrity"] == "PASS"
    assert result["scientific_status"] == "NUMERICALLY VALIDATED / NOT CERTIFIED"
    assert result["canonical_run"] == 45960533214 and result["new_fits"] == 0
    assert result["captured_text_files_verified"] == 12
    assert result["frozen_sources_verified"] == 5
    assert result["coefficient_parity"]["slopes"] < 1e-3
    # Preserve unavailable original-file digests, never reconstruct NPZ ZIP timestamps.
    assert "python_parameters.npz" in result["nonembedded_original_file_digests"]
    assert "integration.csv" in result["nonembedded_original_file_digests"]


def test_actual_raw_archive_is_pinned_to_its_original_committed_blob():
    raw = A.RAW.read_bytes()
    assert hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == A.RAW_GIT_BLOB
    actual = json.loads(raw)
    summary = json.loads(A.REFERENCE.read_text())
    assert actual["report"] == summary["report"]
    assert actual["report"]["calibration_attempts"] == {"Python": 1, "R": 1}


def test_altered_actual_response_bytes_are_refused(tmp_path):
    actual = json.loads(A.RAW.read_text())
    actual["data.csv"] = actual["data.csv"] + "\n"
    damaged = tmp_path / "altered.json"
    damaged.write_text(json.dumps(actual))
    with pytest.raises(AssertionError, match="raw archive blob changed"):
        A.verify(payload_path=damaged)


def test_changed_frozen_scientific_source_is_refused_without_touching_repository(tmp_path):
    for name in A.SOURCES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((A.PROJECT / name).read_bytes())
    damaged = tmp_path / "core/irt_multidim.py"
    damaged.write_bytes(damaged.read_bytes() + b"\n# deliberate test-only tampering\n")
    with pytest.raises(AssertionError, match="Frozen scientific source hash changed"):
        A.verify(project=tmp_path)


def test_verifier_has_no_calibration_download_or_subprocess_path():
    source = Path(A.__file__).read_text()
    tree = ast.parse(source)
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Call):
            function = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            assert function not in {"fit_mirt", "EMEstimator", "r_run", "Popen", "run"}
    assert not modules & {"mirt", "subprocess", "requests", "urllib", "shutil", "pickle"}
    assert "H.inputs" not in source and "download" not in modules
