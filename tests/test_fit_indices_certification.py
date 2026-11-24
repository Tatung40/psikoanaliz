"""LogLik/-2LL/AIC/BIC/parameter counts of all binary models equal independent formulas (fit_indices.json)."""
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_stored_fit_index_evidence_passes_for_every_model():
    ev = json.loads((ROOT / "validation/evidence/binary_certification/fit_indices.json").read_text())
    assert ev["all_pass"] and {r["model"] for r in ev["cases"]} == {"Rasch", "1PL", "2PL", "3PL", "4PL"}


@pytest.mark.parametrize("case", [0, 2])
def test_recomputed_fit_indices_match_independent_formulas(case):
    pytest.importorskip("mirt")
    from validation.binary_certification.fit_indices import CASES, check
    assert check(*CASES[case])["pass"]


def test_infit_outfit_identity_evidence():
    ev = json.loads((ROOT / "validation/evidence/binary_certification/itemfit_identity.json").read_text())
    assert ev["all_pass"] and {r["model"] for r in ev["cases"]} == {"Rasch", "1PL", "2PL", "3PL", "4PL"}


def test_sx2_identity_evidence():
    ev = json.loads((ROOT / "validation/evidence/binary_certification/sx2_identity.json").read_text())
    assert ev["all_pass"] and {r["model"] for r in ev["cases"]} == {"Rasch", "1PL", "2PL", "3PL", "4PL"}


def test_excel_bundle_item_fit_equals_library_itemfit():
    pytest.importorskip("mirt")
    import io
    import pandas as pd
    from core.irt_library_diagnostics import library_itemfit
    from core.irt_python_mirt_adapter import fit_python_mirt
    from core.irt_python_mirt_report import build_report
    from reports.irt_binary_report import bundle
    from validation.binary_certification.common import simulate
    x = simulate("2PL", 1000, 8, 3); names = [f"V{j + 1}" for j in range(8)]
    out = fit_python_mirt(x, model="2PL", n_quadpts=61); rep = build_report(out, model="2PL", items=names)
    itemfit = library_itemfit(out.fit, x, model="2PL")
    persons = pd.DataFrame({"Birey": range(len(x)), "Yetenek (θ)": 0.0, "SH (θ)": 1.0})
    sheets = pd.read_excel(io.BytesIO(bundle(rep, persons, None, "x", itemfit)[0]), sheet_name=None)
    sheet = next(v for k, v in sheets.items() if "fit" in k.lower() or "uyum" in k.lower())
    for col in ("S_X2", "infit", "outfit"):
        assert col in sheet.columns and np.allclose(sheet[col].to_numpy(float), itemfit[col].to_numpy(float), rtol=0, atol=1e-12)
