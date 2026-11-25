"""Recompute archived holdout summaries without fits or fresh random generation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import beta

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("model",["3PL","4PL"])
def test_archived_holdout_evidence_and_conditional_precision(model):
    p=ROOT/"validation/evidence/binary34_person_holdout_f"/f"{model}.json"
    report=json.loads(p.read_text())
    protocol=ROOT/"validation/uncertainty_f/PROTOCOL.json"
    assert hashlib.sha256(protocol.read_bytes()).hexdigest()==report["protocol_sha256"]
    assert report["protocol_commit"]=="72d1d651aa2433fc416cbd960dc7671da9c31e2f"
    assert report["code_commit"]=="44639756faffe1c3baa799a69acc7a29ee6454c9"  # actual pull_request GITHUB_SHA (synthetic merge), not branch head
    assert report["run_id"]=="45593803451"
    assert report["model"]==model and report["fit_converged"] is True
    assert report["calibration_replicates"]==1
    assert report["status"]=="MEASURED / NOT CERTIFIED"
    assert report["certification"]=="NOT CERTIFIED"
    assert report["software_checks"]=="PASS"
    assert report["registry_git_blob"]=="49b5aa914c8ea99ec319ded58bf09ac6b85a4de2"
    summary=report["statistics"]
    n,k=summary["n"],summary["covered"]
    assert n==10000 and 0<=k<=n
    rate=k/n
    assert rate==summary["coverage"]
    assert summary["mcse_conditional_on_calibration"]==pytest.approx(np.sqrt(rate*(1-rate)/n),abs=1e-15)
    expected=[beta.ppf(.0125,k,n-k+1),beta.ppf(.9875,k+1,n-k)]
    np.testing.assert_allclose(summary["cp_97_5"],expected,rtol=0,atol=1e-12)
    assert sum(b["n"] for b in report["theta_bands"])==n
    assert sum(b["covered"] for b in report["theta_bands"])==k
    assert report["quadrature_max_difference"]<=1e-6
    assert report["settings"]["estimation"]=="MAP"
    assert report["settings"]["N_calibration"]==3000
    assert report["settings"]["seed_calibration"]!=report["settings"]["seed_holdout"]
    for key in ("calibration_sha256","holdout_sha256","theta_sha256"):
        assert len(report[key])==64
