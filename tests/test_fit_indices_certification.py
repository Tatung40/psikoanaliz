"""LogLik/-2LL/AIC/BIC/parameter counts of all binary models equal independent formulas (fit_indices.json)."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_stored_fit_index_evidence_passes_for_every_model():
    ev = json.loads((ROOT / "validation/evidence/binary_certification/fit_indices.json").read_text())
    assert ev["all_pass"] and {r["model"] for r in ev["cases"]} == {"Rasch", "1PL", "2PL", "3PL", "4PL"}


@pytest.mark.parametrize("case", [0, 2])
def test_recomputed_fit_indices_match_independent_formulas(case):
    pytest.importorskip("mirt")
    from validation.binary_certification.fit_indices import CASES, check
    assert check(*CASES[case])["pass"]


def test_infit_outfit_identity_evidence():
    ev = json.loads((ROOT / "validation/evidence/binary_certification/itemfit_identity.json").read_text())
    assert ev["all_pass"] and {r["model"] for r in ev["cases"]} == {"Rasch", "1PL", "2PL", "3PL", "4PL"}
