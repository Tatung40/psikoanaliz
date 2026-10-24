"""3PL/4PL usage policy and MAP adapter option (pre-registered evidence, validation/evidence/p1_3pl4pl)."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from core.irt_binary_usage_policy import EVIDENCE, usage_status

ROOT = Path(__file__).resolve().parents[1]


def test_policy_matches_preregistered_evidence_cells():
    summary = json.loads((ROOT / EVIDENCE).read_text(encoding="utf-8"))["summary"]
    for key, cell in summary.items():
        model, n, k, est, theta = key.split("|")
        if theta != "normal" or est not in ("ML21", "MAP"):
            continue
        status, _ = usage_status(model, "ML" if est == "ML21" else "MAP", int(n[1:]), int(k[1:]))
        # "supported" must never be issued for a cell that failed the pre-registered criteria
        if status == "supported":
            assert cell["meets_preregistered_criteria"], key
        if cell["meets_preregistered_criteria"] and est == "MAP":
            assert status == "supported", key


def test_policy_messages_and_scope():
    assert usage_status("2PL", "ML", 500, 10)[0] == "not_applicable"
    assert usage_status("4PL", "MAP", 5000, 8)[0] == "not_supported"
    assert usage_status("3PL", "MAP", 800, 20)[0] == "not_supported"
    assert "N(0,1)" in usage_status("3PL", "MAP", 3000, 20)[1]
    with pytest.raises(ValueError):
        usage_status("3PL", "EAP", 3000, 20)


def test_adapter_map_option_uses_library_default_priors():
    pytest.importorskip("mirt")
    from core.irt_python_mirt_adapter import fit_python_mirt
    rng = np.random.default_rng(3); n, k = 1500, 12
    th = rng.standard_normal(n); a = np.linspace(.8, 1.6, k); b = np.linspace(-1.2, 1.2, k)
    x = (rng.random((n, k)) < .15 + .8 * expit(a * (th[:, None] - b))).astype(int)
    ml = fit_python_mirt(x, model="4PL"); mp = fit_python_mirt(x, model="4PL", estimation="MAP")
    assert ml.estimation == "ML" and mp.estimation == "MAP"
    a_map = mp.fit.coef()["discrimination"].to_numpy()
    assert (a_map < 4.99).all() and not np.allclose(a_map, ml.fit.coef()["discrimination"].to_numpy())
    with pytest.raises(ValueError):
        fit_python_mirt(x, model="2PL", estimation="MAP")
    with pytest.raises(ValueError):
        fit_python_mirt(x, model="3PL", estimation="Bayes")
