"""Committed scientific evidence is complete, internally consistent and matches the reported claims."""
import json
from pathlib import Path

import pytest

EV = Path(__file__).resolve().parents[1] / "validation" / "evidence"


def load(path):
    return json.loads((EV / path).read_text(encoding="utf-8"))


def test_3pl4pl_root_cause_evidence():
    d = load("p1_3pl4pl/irt_3pl4pl_root_cause.json")
    assert d["n_jobs"] == len(d["runs"]) == 880 and all(r["status"] == "MEASURED" for r in d["runs"])
    assert all(r["ll_est_minus_true"] > 0 for r in d["runs"])          # optimizer reached a better maximum
    assert max(r["irf_mapping_maxdiff"] for r in d["runs"]) < 1e-12
    s = d["summary"]
    assert s["4PL|N1000|k12|ML21|normal"]["rmse_a"] > 1.5 and s["4PL|N1000|k12|MAP|normal"]["rmse_a"] < .3
    assert abs(s["4PL|N1000|k12|ML21|normal"]["rmse_a"] - s["4PL|N1000|k12|TRUE0|normal"]["rmse_a"]) < .1
    assert not any(v["meets_preregistered_criteria"] for k, v in s.items() if k.startswith("4PL") and "|ML" in k)


def test_dif_evidence_counts_and_irt_lr_coverage():
    d = load("p2_dif/dif_validation_study.json")
    assert d["n_jobs"] == len(d["runs"]) == 2560
    failed = [r for r in d["runs"] if r["status"] != "OK"]
    assert len(failed) == 1 and "kararlı" in failed[0]["error"]   # fail-closed purification, not hidden
    for row in d["summary"]:
        if row["method"] == "IRTLR":
            assert .925 <= row["coverage_a"] <= .975 and .925 <= row["coverage_b"] <= .975
        if row["kind"] == "ordinal" and row["mu_focal"] == -.5:
            assert row["fwer_holm"] > .075                         # documented inflation under impact


def test_rsm_evidence():
    d = load("p3_rsm/rsm_recovery_study.json")["summary"]
    assert d["library"]["measured"] == 20 and d["library"]["max_abs_dloglik"] < 1e-4 and d["library"]["max_param_diff"] < 1e-3
    for row in d["recovery"]:
        assert row["measured"] == row["attempted"] == 100
        if row["N"] >= 1000:
            for p in ("delta", "tau", "sigma"):
                assert .925 <= row[f"coverage_{p}"] <= .975, (row["N"], row["k"], p)
                assert abs(row[f"bias_{p}"]) <= .05
    for row in d["lr"]:
        assert (row["rejection_rate"] <= .075) if row["truth"] == "RSM" else row["rejection_rate"] > .9


@pytest.mark.parametrize("name", ["polytomous_mc_study.json", "polytomous_mc_study_adapter.json"])
def test_polytomous_evidence(name):
    d = load(f"p4_polytomous/{name}")
    for row in d["summary"]:
        assert row["failed"] == 0 and row["mapping_maxdiff"] < 1e-12 and row["info_rel_maxdiff"] < 1e-4
        if name.endswith("adapter.json") or row["model"] != "NRM":
            assert row["meets_criteria"] or row["N"] != 2000
        else:
            assert row["coverage_slopes"] < .8                      # library-default NRM SE defect (before fix)
