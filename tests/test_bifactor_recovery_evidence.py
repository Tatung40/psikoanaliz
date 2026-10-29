"""Pre-registered bifactor recovery evidence is consistent with its design (the Monte Carlo is not run in CI)."""
import json
from pathlib import Path

from validation.bifactor_recovery_study import CRITERIA, DESIGN, judge

EVIDENCE = Path(__file__).resolve().parents[1] / "validation" / "evidence" / "p8_bifactor" / "recovery.json"


def test_stored_evidence_uses_preregistered_design_and_its_own_judgment():
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["design"] == DESIGN and evidence["criteria"] == CRITERIA
    assert evidence["checks"] == judge(evidence["results"])
    assert evidence["status"] == ("MET" if all(evidence["checks"].values()) else "NOT MET")
    for n in DESIGN["N"]:
        assert evidence["results"][f"N{n}"]["replications"] == DESIGN["replications"]


def test_screen_caption_reports_the_stored_numbers():
    from core.irt_multidim import BIFACTOR_RECOVERY
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "NOT MET" and "ölçüt sağlanmadı" in BIFACTOR_RECOVERY
    for n in DESIGN["N"]:
        r = evidence["results"][f"N{n}"]
        numbers = "/".join(f"{r[k]['rmse']:.3f}".lstrip("0") for k in ("general", "specific", "intercept"))
        assert f"N = {n} {numbers}" in BIFACTOR_RECOVERY
