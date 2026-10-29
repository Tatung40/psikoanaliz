"""Restored B13 output types on the Python mirt route: parameter SE/CI, eight charts, report bundle."""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mirt")
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit

from core import example_datasets as ex
from core.irt_binary_inference import parameter_inference, Z95
from core.irt_person_abilities import person_abilities
from core.irt_python_mirt_adapter import fit_python_mirt
from core.irt_python_mirt_report import build_report
from reports.irt_binary_report import KINDS, kinds, png_bytes, bundle


def _example(model, level=0):
    spec = next(d for d in ex.registry().values() if d.family == "irt" and d.model == model and d.level == level)
    bundle_ = ex.materialize(spec.id)
    return bundle_["data"][bundle_["items"]].astype(int)


def _fit(model, level=0):
    x = _example(model, level)
    outcome = fit_python_mirt(x.to_numpy(), model=model)
    report = build_report(outcome, model=model, items=list(x.columns), theta=np.linspace(-4, 4, 161))
    return x, outcome, report


def _independent_se(x, coef, model, points=21):
    """-Hessian of an independent GH marginal log-likelihood, same 21-node grid as the library fit."""
    nodes, weights = hermegauss(points); logw = np.log(weights / weights.sum())
    y = x.to_numpy(float); k = y.shape[1]
    a0, b0 = coef["discrimination"].to_numpy(float), coef["difficulty"].to_numpy(float)

    def loglik(v):
        a = v[:k] if model == "2PL" else a0
        b = v[k:] if model == "2PL" else v
        p = expit(a * (nodes[:, None] - b))
        z = y @ np.log(p).T + (1 - y) @ np.log1p(-p).T + logw
        top = z.max(1, keepdims=True)
        return float((top[:, 0] + np.log(np.exp(z - top).sum(1))).sum())

    v0 = np.r_[a0, b0] if model == "2PL" else b0
    n, h = len(v0), 1e-4; e = np.eye(n) * h; hess = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            hess[i, j] = hess[j, i] = (loglik(v0 + e[i] + e[j]) - loglik(v0 + e[i] - e[j])
                                       - loglik(v0 - e[i] + e[j]) + loglik(v0 - e[i] - e[j])) / (4 * h * h)
    return np.sqrt(np.diag(np.linalg.inv(-hess)))


@pytest.mark.parametrize("model", ["1PL", "2PL"])
def test_parameter_se_matches_independent_observed_information(model):
    x, outcome, _ = _fit(model)
    table, reason = parameter_inference(outcome.fit, model, x.columns)
    assert reason is None and set(table.Parametre) == ({"b"} if model == "1PL" else {"a", "b"})
    expected = _independent_se(x, outcome.fit.coef(), model)
    shown = np.concatenate([table.loc[table.Parametre == p, "SH"].to_numpy() for p in (["a", "b"] if model == "2PL" else ["b"])])
    np.testing.assert_allclose(shown, expected, rtol=1e-3)
    np.testing.assert_allclose(table["GA %95 üst"] - table["Kestirim"], Z95 * table["SH"], rtol=1e-12)
    assert table.Madde.tolist()[:1] == [x.columns[0]]


@pytest.mark.parametrize("model", ["3PL", "4PL"])
def test_boundary_fits_report_reason_instead_of_fabricated_se(model):
    x, outcome, _ = _fit(model)
    table, reason = parameter_inference(outcome.fit, model, x.columns)
    assert table is None and "sonlu değil" in reason


def test_nonconverged_fit_has_no_se():
    x = _example("2PL")
    outcome = fit_python_mirt(x.to_numpy(), model="2PL", max_iter=1)
    assert parameter_inference(outcome.fit, "2PL", x.columns) == (None, "Kestirim yakınsamadı; SH ve Wald GA üretilmedi.")


@pytest.mark.parametrize("model", ["2PL", "4PL"])
def test_every_chart_kind_is_a_300_dpi_png_and_bundle_is_complete(model):
    x, outcome, report = _fit(model)
    persons = person_abilities(x.to_numpy(), report.curves, model=model)
    expected = KINDS if model == "4PL" else KINDS[:-1]
    assert kinds(model) == expected
    for kind in expected:
        data = png_bytes(report, persons, kind)
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
    if model == "2PL":
        with pytest.raises(ValueError):
            png_bytes(report, persons, "Asimptotlar")
    from PIL import Image
    assert round(Image.open(io.BytesIO(png_bytes(report, persons, "TIF"))).info["dpi"][0]) == 300
    inference, reason = parameter_inference(outcome.fit, model, x.columns)
    xlsx, archive = bundle(report, persons, inference, reason)
    sheets = pd.read_excel(io.BytesIO(xlsx), sheet_name=None)
    assert {"Model_Ozeti", "Madde_Parametreleri", "Kisi_Yetenekleri", "Bilgi_SEM"} <= set(sheets)
    assert ("Parametre_SH_GA" in sheets) == (inference is not None)
    names = zipfile.ZipFile(io.BytesIO(archive)).namelist()
    assert {f"grafikler/{k}.png" for k in expected} <= set(names) and f"{model}_yontem.txt" in names
    np.testing.assert_allclose(sheets["Madde_Parametreleri"]["b"], report.parameters["b"], rtol=0, atol=1e-12)


def test_library_screen_exposes_extras_and_invalidates_bundle_on_data_change():
    from streamlit.testing.v1 import AppTest
    x = _example("2PL")
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=240)
    for key, value in dict(raw_df=x, scored_df=x.copy(), item_cols=list(x.columns), id_col=None, group_col=None,
                           is_dichotomous=True, nav_page="MTK: Python mirt 2PL").items():
        at.session_state[key] = value
    at.run()
    at.button(key="python_mirt_start_2PL").click().run()
    assert not at.exception
    assert not any("SH ve %95 Wald" in m.value for m in at.markdown)  # opt-in, like the Rasch Wright map
    at.checkbox(key="python_mirt_extras_2PL").check().run()
    assert not at.exception
    assert any("SH ve %95 Wald" in m.value for m in at.markdown)
    shown = at.dataframe[-1].value
    assert list(shown.columns)[:4] == ["Madde", "Parametre", "Kestirim", "SH"] and len(shown) == 2 * x.shape[1]
    at.button(key="python_mirt_bundle_run_2PL").click().run()
    assert not at.exception
    assert at.session_state["python_mirt_bundle"]["2PL"]["files"][1][:2] == b"PK"
    changed = x.copy(); changed.iloc[0, 0] = 1 - changed.iloc[0, 0]
    at.session_state["scored_df"] = changed
    at.run()
    assert not at.exception
    assert not any("SH ve %95 Wald" in m.value for m in at.markdown)
