"""Replay bundles must carry every non-.py file the called calculation modules read at runtime."""
import io, json, subprocess, sys, zipfile
from core.reproducible import runtime_hashes, bundle, python_code


def test_assets_by_called_module():
    h = runtime_hashes([{"module": "core.rater_reliability", "function": "compute"}])
    assert {"core/rater_methods.json", "validation/rater_reliability/evidence.json"} <= set(h)
    assert "scripts/mfrm_tam.R" in runtime_hashes([{"module": "core.mfrm", "function": "fit"}])
    assert "validation/lord_rasch_free_scale/results/report.json" in runtime_hashes([{"module": "core.dif_lord_rasch", "function": "x"}])
    assert "validation/evidence/binary_certification/CERTIFICATION.json" in runtime_hashes([{"module": "core.irt_binary_inference", "function": "x"}])
    plain = runtime_hashes([{"module": "core.descriptives", "function": "x"}])
    assert not any(k.endswith((".json", ".R", ".csv")) for k in plain)


def test_rater_replay_executes_from_bundle_with_identical_result(tmp_path):
    import os
    import pandas as pd
    import core.rater_reliability as R
    from core.reproducible import begin, finish, public_call
    frame = pd.DataFrame({"id": range(8), "r1": [1, 2, 3, 1, 2, 3, 1, 2], "r2": [1, 2, 3, 2, 2, 3, 1, 1]})
    rec = begin("Puanlayıcı Güvenirliği")
    try:
        data = public_call("core.rater_reliability", "from_wide", frame, "id", ["r1", "r2"])
        expected = public_call("core.rater_reliability", "compute", "cohen_kappa", data)
    finally:
        finish()
    manifest = rec.manifest(); code = python_code(manifest)
    assert "core/rater_methods.json" in manifest["runtime_sha256"]
    with zipfile.ZipFile(io.BytesIO(bundle(manifest, code))) as archive:
        archive.extractall(tmp_path)
    assert (tmp_path / "runtime/core/rater_methods.json").is_file() and (tmp_path / "runtime/validation/rater_reliability/evidence.json").is_file()
    script = tmp_path / "analysis.py"
    script.write_text(code + "\nimport json as _j; open('out.json','w').write(_j.dumps(results[-1], default=float))\n")
    run = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=dict(os.environ, PYTHONPATH=""), capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr[-2000:]
    got = json.loads((tmp_path / "out.json").read_text())
    assert abs(got["estimate"] - expected["estimate"]) <= 1e-12
