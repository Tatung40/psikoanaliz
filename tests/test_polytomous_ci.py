import pytest
from scripts.dif_ci_paths import choose_polytomous


@pytest.mark.parametrize("path,expected", [
    ("core/polytomous_dif.py", True), ("reports/polytomous_dif_report.py", True),
    ("core/grm_dif.py", True), ("core/irt_dif.py", True), ("core/rasch.py", True),
    ("scripts/rsm_dif_reference.py", True), ("tests/fixtures/polytomous_dif/PCM_simulation.json", True),
    ("app.py", True), ("core/irt_mg34.py", False), ("core/ctt.py", False),
])
def test_relevant_polytomous_gate(path, expected):
    assert choose_polytomous([path]) is expected
