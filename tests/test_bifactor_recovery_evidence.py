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
