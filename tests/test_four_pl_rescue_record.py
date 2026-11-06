"""4PL rescue record (validation/four_pl_rescue): preregistration precedes results; failures stay; production gate unchanged."""
import hashlib, json, subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / "validation/four_pl_rescue"


def _sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def test_fixed_designs_match_preregistered_hashes():
    P = json.loads((H / "PROTOCOL.json").read_text())
    assert _sha(P["designs"]["A"]["data"]) == P["designs"]["A"]["sha256"]
    assert _sha(P["designs"]["B"]["data"]) == P["designs"]["B"]["sha256"]
    assert _sha(P["designs"]["B"]["truth"]) == P["designs"]["B"]["truth_sha256"]
    assert P["numerical_gates"]["projected_score"] == 1e-5 and P["numerical_gates"]["condition"] == 1e8


def test_verdicts_record_failures_and_block_switch():
    for d in ("A", "B"):
        v = json.loads((H / f"verdict_{d}.json").read_text())
        assert v["protocol_commit"] == "dd3d646" and not v["numerical_pass"] and "recovery" not in v
    assert json.loads((H / "verdict_A.json").read_text())["classification"] == "OUTSIDE COMMON BOUNDS"
    import core.irt_model_comparison as C
    assert C.FOUR_PL_ML_CERTIFIED_SCOPE is None and C.four_pl_gate(10000, 30, 0)[0] is False


def test_protocol_committed_before_verdicts():
    try:
        log = lambda f: subprocess.run(["git", "log", "--format=%ct", "--diff-filter=A", "--", f], cwd=ROOT,
                                       capture_output=True, text=True, check=True).stdout.split()
        proto, verdict = log("validation/four_pl_rescue/PROTOCOL.json"), log("validation/four_pl_rescue/verdict_B.json")
    except Exception:
        pytest.skip("git history unavailable")
    if not proto or not verdict:
        pytest.skip("shallow clone")
    assert int(proto[-1]) <= int(verdict[-1])
