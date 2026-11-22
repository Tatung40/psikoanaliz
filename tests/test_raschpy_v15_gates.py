"""V15 scientific gates must fail closed on incomplete simulation evidence."""
import numpy as np
import pytest
from validation.raschpy_mfrm.simulation_v15 import summarise


def _row(rep, kind="null"):
    row = {"kind": kind, "rep": rep}
    for test in ("dif", "drf"):
        row.update({f"{test}_rej": 1, f"{test}_n": 20,
                    f"{test}_flag_any": False, f"{test}_om_rej": False,
                    f"{test}_identity": {"z": 0., "p": 0., "p_bh": 0.}})
    if kind == "power":
        row["k3_flagged"] = True
    return row


def test_v15_complete_data_keeps_identity_gate_available():
    out = summarise([_row(i) for i in range(60)] +
                    [_row(i, "power") for i in range(30)])
    for t in ("dif", "drf"):
        assert out[t]["complete"] is True
        assert out[t]["gate_identity"] is True
    assert out["dif"]["power_complete"] is True


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nan", "missing_omnibus"])
def test_v15_incomplete_or_nonfinite_evidence_never_passes(mutation):
    rows = [_row(i) for i in range(60)] + [_row(i, "power") for i in range(30)]
    if mutation == "missing":
        rows.pop(0)
    elif mutation == "duplicate":
        rows[0]["rep"] = 1
    elif mutation == "nan":
        rows[0]["dif_identity"]["p"] = np.nan
    else:
        rows[0]["drf_om_rej"] = None
    out = summarise(rows)
    for t in ("dif", "drf"):
        if mutation == "nan" and t == "drf":
            continue
        if mutation == "missing_omnibus" and t == "dif":
            continue
        assert out[t]["complete"] is False
        assert not any(out[t][k] for k in ("gate_identity", "gate_size", "gate_omnibus"))


def test_v15_missing_power_is_disclosed():
    out = summarise([_row(i) for i in range(60)])
    assert out["dif"]["power_complete"] is False
    assert out["dif"]["power_k3_flagged"] is None
