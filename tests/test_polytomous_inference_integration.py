"""P4-C1 production integration: shared table labels, fail-closed scope, XLSX/ZIP identical to the screen table."""
import io, sys, zipfile
import numpy as np
import pandas as pd
import pytest
from core.irt_python_mirt_adapter import fit_python_mirt
from core.polytomous_inference import parameter_table
sys.path.insert(0, "validation")
import polytomous_mc_study as S

ITEMS = [f"m{j}" for j in range(S.K)]


@pytest.fixture(scope="module")
def fits():
    out = {}
    for m in ("GRM", "PCM", "GPCM", "NRM"):
        x, _ = S.simulate(m, 2000, 777000 + len(out))
        out[m] = fit_python_mirt(x, model=m, n_categories=S.C)
    return out


def test_seven_blocks_certified_in_scope_and_nrm_oakes_recorded(fits):
    blocks = set()
    for m, o in fits.items():
        t, reason = parameter_table(o, ITEMS); assert reason is None
        assert t["Bilimsel durum (GA)"].str.startswith("SCOPED CERTIFIED").all(), m
        blocks |= {(m, b) for b in t["Parametre"].unique()}
    assert fits["NRM"].se_method == "oakes" and fits["GRM"].se_method == "library_default"
    assert blocks == {("GRM", "discrimination"), ("GRM", "thresholds"), ("PCM", "steps"), ("GPCM", "discrimination"),
                      ("GPCM", "steps"), ("NRM", "slopes"), ("NRM", "intercepts")}


def test_fail_closed_outside_scope(fits):
    import dataclasses
    for change in ({"n_persons": 500}, {"se_method": None}, {"solver_settings": {"max_iter": 500, "tol": 1e-4, "n_quadpts": 61}},
                   {"estimation": "MAP"}, {"n_items": 12}):
        o = dataclasses.replace(fits["NRM"], **change)
        assert (parameter_table(o, ITEMS + ["x", "y"])[0]["Bilimsel durum (GA)"] == "NOT CERTIFIED").all(), change
    assert parameter_table(dataclasses.replace(fits["GRM"], model="RSM"), ITEMS)[0] is None


def test_xlsx_and_zip_carry_identical_table(fits):
    from core.polytomous_inference import export_bundle
    o = fits["GPCM"]; t, _ = parameter_table(o, ITEMS)
    excel, archive = export_bundle(t)
    sheet = pd.read_excel(io.BytesIO(excel), sheet_name="Parametre_SH_GA")
    assert sheet["Bilimsel durum (GA)"].tolist() == t["Bilimsel durum (GA)"].tolist()
    assert np.allclose(sheet["%95 GA alt"], t["%95 GA alt"])
    z = zipfile.ZipFile(io.BytesIO(archive)); assert pd.read_csv(z.open("Parametre_SH_GA.csv"))["Bilimsel durum (GA)"].tolist() == t["Bilimsel durum (GA)"].tolist()
