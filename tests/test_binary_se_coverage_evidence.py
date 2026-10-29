"""CI gate for the pre-registered 1PL/2PL SE/CI coverage evidence (the Monte Carlo itself is not run in CI)."""
import json
from pathlib import Path

from validation.binary_se_coverage_study import CRITERIA, DESIGN, judge

EVIDENCE = Path(__file__).resolve().parents[1] / "validation" / "evidence" / "p6_binary_se" / "coverage.json"


def test_stored_evidence_uses_the_preregistered_design_and_meets_its_criteria():
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["design"] == DESIGN and evidence["criteria"] == CRITERIA
    assert evidence["checks"] == judge(evidence["results"]) and all(evidence["checks"].values())
    for model in DESIGN["models"]:
        result = evidence["results"][model]
        assert result["replications"] == DESIGN["replications"] and result["se_withheld"] == 0
        for name in ("a", "b") if model == "2PL" else ("b",):
            assert result[name]["intervals"] == DESIGN["replications"] * DESIGN["items"]
