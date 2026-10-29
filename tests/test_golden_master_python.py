"""Python-library outputs must match the pre-removal golden master (native engine excluded).

Master: validation/golden_master/python_outputs.json.gz, written by tools/golden_master.py on
main BEFORE the native binary IRT engine was removed. Predeclared tolerance (not tuned to
results): rtol 1e-6, atol 1e-8 for floats; every string, label, shape and order is exact.
Locally the replay is bit-identical; the tolerance only absorbs cross-platform BLAS noise.
"""
import pytest

pytest.importorskip("mirt")
pytest.importorskip("generalizit")
from tools import golden_master as gm

MASTER = gm.load()


@pytest.mark.parametrize("section", sorted(gm.SECTIONS))
def test_python_outputs_match_golden_master(section):
    actual = gm.run_section(section)
    differences = gm.compare(MASTER[section], actual, section, rtol=1e-6, atol=1e-8)
    assert not differences, "\n".join(differences[:20])


def test_master_covers_every_section():
    assert set(MASTER) == set(gm.SECTIONS)
