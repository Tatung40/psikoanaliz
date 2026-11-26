"""Declared metadata-only source changes must reverse exactly to the frozen q1_q3 historical digest."""
import hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def test_declared_source_changes_reverse_to_historical_digest():
    m = json.loads((ROOT / "validation/q1_q3/manifest.json").read_text(encoding="utf-8"))
    assert m["source_changes"]
    for path, c in m["source_changes"].items():
        text = (ROOT / path).read_text(encoding="utf-8")
        assert hashlib.sha256(text.encode()).hexdigest() == c["current_sha256"], path
        r = c["reversal"]; assert text.count(r["remove"]) == 1 and text.count(r["replace_new"]) == 1
        old = text.replace(r["remove"], "", 1).replace(r["replace_new"], r["with_old"], 1)
        assert hashlib.sha256(old.encode()).hexdigest() == c["historical_sha256"], path
        assert path not in m[c["moved_from_family"]]
