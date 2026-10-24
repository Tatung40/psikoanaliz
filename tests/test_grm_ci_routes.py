import pytest
from scripts.dif_ci_paths import choose


@pytest.mark.parametrize("paths,expected", [
    # Shared app navigation executes binary DIF too; enforce its regression gate.
    (["core/grm_dif.py", "reports/grm_dif_report.py", "app.py"], dict(binary=True, grm=True)),
    (["core/grm_dif.py", "reports/grm_dif_report.py"], dict(binary=False, grm=True)),
    (["app.py"], dict(binary=True, grm=True)),
    (["core/irt_dif.py"], dict(binary=True, grm=True)),
    (["core/rasch.py"], dict(binary=True, grm=True)),
    (["core/irt_mg34.py"], dict(binary=True, grm=False)),
    (["tests/test_dif_fixed_anchors.py"], dict(binary=True, grm=False)),
    (["ui/irt_b10_view.py"], dict(binary=True, grm=False)),
    (["tests/fixtures/grm_dif/python_mirt_null.json"], dict(binary=False, grm=True)),
    (["core/ctt.py"], dict(binary=False, grm=False)),
])
def test_ci_keeps_shared_and_binary_gates_without_repeating_unaffected_binary(paths, expected):
    assert choose(paths) == expected
