"""B14 completion: persons, information (independent finite-difference check), model comparison, exports."""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mirt")
from core.irt_b14 import fit_b14
from core.irt_b14_persons_info import independent_information, information_table, model_comparison, person_scores
from reports.irt_b14_report import export_xlsx, export_zip, report_tables
from validation.polytomous_mc_study import simulate


@pytest.fixture(scope="module", params=["GRM", "PCM", "GPCM", "NRM"])
def fitted(request):
    x, _ = simulate(request.param, 800, 5)
    if request.param == "NRM":
        frame = pd.DataFrame(x).replace({0: "A", 1: "B", 2: "C", 3: "D"})
        return fit_b14(frame, "NRM", categories=("A", "B", "C", "D"))
    return fit_b14(pd.DataFrame(x, columns=[f"P{j}" for j in range(x.shape[1])]), request.param)


def test_information_matches_independent_fisher_information(fitted):
    th = np.linspace(-3, 3, 25)
    table = information_table(fitted, th)
    lib = table.filter(like="I_").to_numpy()
    ind = independent_information(fitted, th)
    assert np.allclose(lib, ind, rtol=1e-4, atol=1e-8)
    assert np.allclose(table["Test bilgisi"], lib.sum(1)) and np.allclose(table["SEM(θ)"] ** -2, table["Test bilgisi"])


def test_person_scores_are_finite_and_aligned(fitted):
    p = person_scores(fitted)
    assert len(p) == fitted["N"] and np.isfinite(p["θ"]).all() and (p["SH(θ)"] > 0).all()
    assert ("Ham puan" in p) == (fitted["model"] != "NRM")


def test_reports_include_persons_and_information(fitted):
    frames = report_tables(fitted)
    assert {"Kişi Yetenekleri", "Bilgi Fonksiyonları"} <= set(frames)
    with zipfile.ZipFile(io.BytesIO(export_zip(fitted))) as z:
        assert "kisi_yetenekleri.csv" in z.namelist() and "bilgi_fonksiyonlari.csv" in z.namelist()
    pd.read_excel(io.BytesIO(export_xlsx(frames)), sheet_name="Kişi Yetenekleri")


def test_model_comparison_nested_lr_only_for_pcm_gpcm():
    x, _ = simulate("GPCM", 1500, 6)
    table, lr = model_comparison(x)
    assert list(table["Model"]) == ["GRM", "PCM", "GPCM"] and table["Yakınsama"].all()
    assert lr is not None and lr["sd"] == 10 and lr["p"] < .05  # true slopes vary
    assert table["En küçük AIC"].sum() == 1
