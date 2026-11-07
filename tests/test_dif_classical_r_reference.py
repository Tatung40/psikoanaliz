"""Observed-score DIF vs independent R (validation/dif_classical_r/PROTOCOL.json); purify=False for identical matching."""
import json, os, shutil, subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.dif_classical import dif_screen

ROOT = Path(__file__).resolve().parents[1]; H = ROOT / "validation/dif_classical_r"
TOL = {"chi2": 1e-9, "alpha_mh": 1e-9, "delta": 1e-9, "lnOR %95 alt": 1e-8, "lnOR %95 üst": 1e-8, "smd": 1e-10}


def tol(kind, key, value):
    if key.startswith("p"): return 1e-9 + 1e-6 * abs(value)
    if kind == "ordinal" and key.startswith("lr_"): return 1e-3
    if kind == "ordinal" and key.startswith("delta_r2"): return 1e-5
    if kind == "ordinal" and key == "chi2": return 1e-10
    return TOL.get(key, 1e-6)


def ours():
    P = json.loads((H / "PROTOCOL.json").read_text())
    b = pd.read_csv(ROOT / P["data"]["binary"]["path"]); o = pd.read_csv(ROOT / P["data"]["ordinal"]["path"])
    out = {}
    for kind, df, grp, ref, foc, ordinal in (("binary", b, "Grup", "R", "F", False), ("ordinal", o, "grup", "Referans", "Odak", True)):
        items = [c for c in df.columns if c != grp]
        mh, _ = dif_screen(df[items], df[grp], ref, foc, method="MH", ordinal=ordinal, purify=False)
        lr, _ = dif_screen(df[items], df[grp], ref, foc, method="LR", ordinal=ordinal, purify=False)
        out[kind] = mh.set_index("Madde").join(lr.set_index("Madde")[[c for c in lr.columns if c.startswith(("lr_", "delta_r2")) or
                                                                   c in ("p_uniform", "p_nonuniform", "p_omnibus")]], how="left")
    return out


def compare(ref_csv):
    ref = pd.read_csv(ref_csv); mine = ours(); rows = []
    for r in ref.itertuples():
        got = float(mine[r.kind].loc[r.item, r.key]); t = tol(r.kind, r.key, r.value)
        rows.append(dict(kind=r.kind, item=r.item, key=r.key, python=got, r=r.value, diff=abs(got - r.value), tol=t,
                         ok=bool(abs(got - r.value) <= t)))
    return pd.DataFrame(rows)


def test_observed_score_dif_matches_archived_independent_r():
    d = compare(H / "reference.csv")
    assert len(d) == 256 and d.ok.all(), d[~d.ok].to_string()


def test_rerun_r_reference_when_available(tmp_path):
    rs = shutil.which("Rscript")
    if not rs:
        if os.getenv("REQUIRE_DIF_R") == "1": pytest.fail("R required")
        pytest.skip("NOT EXECUTED: Rscript unavailable")
    out = tmp_path / "reference.csv"; P = json.loads((H / "PROTOCOL.json").read_text())
    run = subprocess.run([rs, str(H / "reference.R"), str(ROOT / P["data"]["binary"]["path"]), str(ROOT / P["data"]["ordinal"]["path"]), str(out)],
                         capture_output=True, text=True, timeout=200)
    if run.returncode != 0 and "MASS" in run.stderr and os.getenv("REQUIRE_DIF_R") != "1":
        pytest.skip("NOT EXECUTED: MASS unavailable")
    assert run.returncode == 0, run.stderr[-1500:]
    d = compare(out); d.to_csv(tmp_path / "comparison.csv", index=False); assert d.ok.all()
