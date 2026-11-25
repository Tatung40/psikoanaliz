"""Independently verify archived scientific FAIL/INCONCLUSIVE evidence; no fitting."""
import hashlib,json
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
CASES=[(m,n,j) for m in ("3PL","4PL") for n in (3000,10000) for j in (12,30)]


@pytest.mark.parametrize("model,n,j",CASES,ids=[f"{m}-{n}-{j}" for m,n,j in CASES])
def test_archived_confirmation(model,n,j):
    r=json.loads((ROOT/"validation/evidence/parameter_uncertainty_g"/f"{model}_{n}_{j}.json").read_text())
    protocol=(ROOT/"validation/parameter_uncertainty_g/PROTOCOL.json").read_bytes()
    assert r["protocol_sha256"]==hashlib.sha256(protocol).hexdigest()
    assert r["protocol_commit"]=="7c6dba3ccfa9475260053938e997f0c5d57e6078"
    assert r["run_id"]=="45606844841"
    assert (r["model"],r["N"],r["J"])==(model,n,j)
    assert r["certification"]=="NOT CERTIFIED"
    assert r["planned"]==(300 if n==3000 else 150)
    assert r["valid"]+r["se_refused"]==r["attempted"]<=r["planned"]
    assert r["checks"]["complete"]==(r["attempted"]==r["planned"])
    assert r["cell_pass"]==all(r["checks"].values())
    for q,s in r["parameter"].items():
        a=np.array(s["coverage_cluster_counts"]["covered"],float)
        b=np.array(s["coverage_cluster_counts"]["eligible"],float)
        assert len(a)==len(b)==r["attempted"]
        assert np.all((a>=0)&(a<=b))
        p=a.sum()/b.sum()
        assert p==pytest.approx(s["coverage"],abs=1e-15)
        influence=a-p*b
        mcse=np.sqrt(len(a)/(len(a)-1)*np.dot(influence,influence))/b.sum()
        assert mcse==pytest.approx(s["coverage_mc_se"],abs=1e-15)
        assert r["checks"][q+"_coverage"]==(abs(p-.95)<=.02)
        assert r["checks"][q+"_precision"]==(mcse<=.005)
        assert s["failed_as_noncoverage"]==pytest.approx(a.sum()/(r["attempted"]*j))
        assert len(s["per_item_coverage"])==j
