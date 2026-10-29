"""Bifactor teaching datasets must be analysable with the MIRT screen's default settings."""
import numpy as np
import pytest

pytest.importorskip("mirt")
from core import example_datasets as ex
from core.irt_multidim import fit, report


@pytest.mark.parametrize("dataset_id", [d.id for d in ex.registry().values() if d.model == "Bifactor"])
def test_bifactor_example_fits_with_screen_defaults(dataset_id):
    spec = ex.registry()[dataset_id]
    data = ex.materialize(dataset_id)
    items = data["items"]
    # ui/irt_multidim_view.py defaults: specific group i*(dims-1)//k, 5 GH nodes for 4 dimensions, 1000 iterations.
    groups = [i * max(spec.dimensions - 1, 1) // len(items) for i in range(len(items))]
    assert spec.dimensions == 4 and len(set(groups)) == 3 and min(np.bincount(groups)) >= 4
    result = report(fit(data["data"][items], "Bifactor", groups, n_quadpts=5, max_iter=1000), data["data"][data["id"]].to_numpy())
    assert result["metadata"]["Converged"] is True
