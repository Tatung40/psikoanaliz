"""Rasch routing regression: no new estimator, only the fixed-a 1PL alias."""
from types import SimpleNamespace
import numpy as np
from core import irt_python_mirt_adapter as adapter


def test_rasch_routes_to_fixed_discrimination_library_1pl(monkeypatch):
    calls = []
    fake = SimpleNamespace(fit_mirt=lambda x, **kw: (calls.append(kw) or object()))
    monkeypatch.setattr(adapter, "import_module", lambda name: fake)
    x = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1]])
    result = adapter.fit_python_mirt(x, model="Rasch")
    assert result.model == "Rasch"
    assert calls == [{"model": "1PL", "max_iter":500, "tol":1e-4, "n_quadpts":21}]
    assert result.solver_settings == {"max_iter":500, "tol":1e-4, "n_quadpts":21}


def test_free_shared_discrimination_stays_separate(monkeypatch):
    calls = []
    fake = SimpleNamespace(fit_mirt=lambda x, **kw: (calls.append(kw) or object()))
    monkeypatch.setattr(adapter, "import_module", lambda name: fake)
    x = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1]])
    adapter.fit_python_mirt(x, model="1PL", common_discrimination=True)
    assert calls[0]["model"] == "2PL"
    assert calls[0]["constraints"] == [{"parameter": "discrimination", "items": [0, 1, 2]}]
