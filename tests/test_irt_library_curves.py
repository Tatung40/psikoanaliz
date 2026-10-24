import numpy as np
import pytest
from core.irt_library_curves import binary_curves


def test_rasch_midpoint_and_information():
    r = binary_curves(np.array([-1., 0., 1.]), a=[1.], b=[0.])
    assert r["probability"][1, 0] == pytest.approx(.5)
    assert r["information"][1, 0] == pytest.approx(.25)


def test_4pl_asymptotes_and_nonnegative_information():
    r = binary_curves(np.array([-100., 0., 100.]),
                      a=[1.5], b=[0.], c=[.2], d=[.9])
    assert r["probability"][0, 0] == pytest.approx(.2)
    assert r["probability"][-1, 0] == pytest.approx(.9)
    assert np.all(r["information"] >= 0)


def test_test_information_is_sum_of_items():
    r = binary_curves(np.linspace(-3, 3, 15),
                      a=[1., 1.3], b=[-.5, .5])
    np.testing.assert_allclose(r["test_information"], r["information"].sum(axis=1))


@pytest.mark.parametrize("kwargs", [
    dict(a=[0.], b=[0.]),
    dict(a=[1.], b=[0.], c=[.8], d=[.7]),
    dict(a=[1.], b=[0.], c=[-.1]),
])
def test_invalid_parameters_fail_closed(kwargs):
    with pytest.raises(ValueError):
        binary_curves(np.array([0.]), **kwargs)
