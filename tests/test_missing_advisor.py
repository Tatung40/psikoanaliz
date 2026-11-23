"""Central missing-data advisor: deterministic, supported methods only, no silent imputation/deletion, IRT hand-off."""
import numpy as np
import pandas as pd
import pytest

import core.missing_advisor as A
from core.missing_imputation import METHODS, apply

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def binary(n=400, k=8, rate=0., seed=1):
    r = np.random.default_rng(seed); th = r.normal(size=n)
    x = pd.DataFrame((r.random((n, k)) < 1 / (1 + np.exp(-(th[:, None] - np.linspace(-1, 1, k))))).astype(float),
                     columns=[f"V{j + 1}" for j in range(k)])
    if rate: x = x.mask(r.random(x.shape) < rate)
    return x


def test_no_missing_and_native_engines_never_impute():
    x = binary(); assert A.advise(x, list(x), "MTK: 2PL")["status"] == "none"
    y = binary(rate=.1); a = A.advise(y, list(y), "MTK: RaschPy MFRM")
    assert a["status"] == "native" and a["recommended"] == "keep" and set(a["allowed"]) == {"keep"}


@pytest.mark.parametrize("page", ["MTK: Python mirt 1PL", "MTK: Python mirt 4PL", "MTK: 2PL", "MTK: Karşılaştırma"])
def test_binary_irt_no_primary_imputation_assumptions_stated_category_safe(page):
    for x in (binary().mask(binary().index.to_series().eq(3).to_frame().reindex(columns=binary().columns, fill_value=False)
                            .assign(V4=lambda f: f.index == 3)), binary(rate=.05)):
        a = A.advise(x, list(x), page)
        assert a["status"] == "action" and a["recommended"] is None and A.engine(page)["native"] is False
        assert "önerilen birincil yöntem yoktur" in a["reason"]                     # no primary method exists
        assert "MCAR" in a["allowed"]["listwise"] and "tek başına gerekçe değildir" in a["allowed"]["listwise"]
        assert a["allowed"]["mode"].startswith("Duyarlılık analizi seçeneği — potansiyel olarak yanlı")
        assert "median" in a["excluded"] and "keep" in a["excluded"] and set(a["allowed"]) <= set(METHODS)
        assert A.advise(x, list(x), page) == a                                       # deterministic
    heavy = binary(rate=.05); out = apply(heavy, list(heavy), "mode")
    assert set(pd.unique(out.to_numpy().ravel())) <= {0., 1.}                       # categories preserved
    obs = heavy.notna(); assert ((out == heavy) | ~obs).all().all()                 # observed values untouched


def test_engine_audit_registers_verified_ctt_rules():
    import core.scoring as S, core.reliability as R
    x = binary(); x.iloc[[1, 5, 9], 3] = np.nan
    tot = S.calculate_total_scores(x, list(x))
    assert tot.iloc[1] == x.iloc[1].sum() and A.engine("4. Test İstatistikleri")["native"] is False
    assert "%80" in A.engine("5. Madde Analizi")["note"]
    assert R.compute_reliability_indices(x, list(x))["n_examinees"] == 397 and "tam kayıt" in A.engine("7. Güvenirlik & Ölçme Hatası")["note"]
    assert A.engine("Bilinmeyen Analiz")["native"] is None                            # unverified stays unregistered
    assert A.engine("4. Test İstatistikleri")["approval"] and A.engine("7. Güvenirlik")["approval"]


def test_listwise_text_states_mcar_sufficient_not_necessary():
    t = A.LISTWISE
    assert "yeterli" in t and "gerekli değildir" in t and "yalnız MCAR" not in t


def test_registered_engine_probes():
    """Small probes behind each registry entry: native engines run with NaN; refusing engines refuse."""
    import core.irt_rsm as R, core.grm_dif as G, core.dimensionality as D, core.dif_classical as C
    import core.irt_multidim as M
    rng = np.random.default_rng(7)
    x = pd.DataFrame(R.simulate(np.linspace(-1, 1, 4), np.array([-.8, .8]), 300, seed=5), columns=list("abcd")).astype(float)
    x.iloc[:20, 1] = np.nan; g = np.r_[np.zeros(150), np.ones(150)]
    assert R.fit(x)["converged"] and R.prepare(x)["missing"] == 20 and A.engine("MTK: RSM")["native"] is True
    assert isinstance(G.test_dif(x, g, 0, 1, anchors=["a", "c"]), dict) and A.engine("MTK: GRM Formel DIF")["native"] is True
    b = pd.DataFrame((rng.random((300, 5)) < .6).astype(float), columns=list("abcde")); b.iloc[:20, 1] = np.nan
    n = D.correlation_matrix(b)["n_pairs"]; assert n.iloc[0, 1] == 280 and n.iloc[0, 2] == 300      # pairwise N
    assert A.engine("Boyutluluk ve Faktör Analizi")["approval"]
    with pytest.raises(ValueError, match="eksiksiz"):
        C.dif_screen(b, g, 0, 1)
    with pytest.raises(ValueError, match="eksiksiz"):
        M.fit(pd.concat([b, b.add_prefix("z")], axis=1).iloc[:, :6], "2PL", np.c_[np.r_[1, 1, 1, 0, 0, 0], np.r_[0, 0, 0, 1, 1, 1]])
    for page in ("MTK: Gözlenen Puan DIF", "Çok Boyutlu MTK (MIRT)", "Yerel Bağımsızlık ve Madde Uyumu"):
        assert A.engine(page)["native"] is False


def test_export_provenance_stamping():
    import io, zipfile
    from docx import Document
    from ui.missing_advisor_view import stamp
    rec = dict(method="listwise", columns_n=8, cells=2, n_before=400, n_after=398, page="MTK: Python mirt 2PL")
    csv = stamp(b"\xef\xbb\xbfa,b\n1,2\n", "t.csv", rec)
    assert csv == b"\xef\xbb\xbfa,b\n1,2\n"                                          # standard CSV, no comment lines
    xb = io.BytesIO(); pd.DataFrame({"a": [1]}).to_excel(xb, index=False)
    assert "Kayıp veri işlemi" in pd.read_excel(io.BytesIO(stamp(xb.getvalue(), "t.xlsx", rec)), sheet_name=None)
    db = io.BytesIO(); Document().save(db)
    assert "Tam kayıt" in "\n".join(p.text for p in Document(io.BytesIO(stamp(db.getvalue(), "t.docx", rec))).paragraphs)
    zb = io.BytesIO(); zipfile.ZipFile(zb, "w").close()
    assert "kayip_veri_islemi.txt" in zipfile.ZipFile(io.BytesIO(stamp(zb.getvalue(), "t.zip", rec))).namelist()
    assert stamp(b"x", "t.png", rec) == b"x"


def test_ordinal_continuous_and_all_missing_rules():
    r = np.random.default_rng(2); o = pd.DataFrame(r.integers(1, 6, (300, 5)).astype(float), columns=list("abcde")).mask(r.random((300, 5)) < .05)
    a = A.advise(o, list(o), "Bilinmeyen analiz")
    assert a["engine"]["native"] is None and a["recommended"] is None and {"median", "knn", "zero"} <= set(a["excluded"])
    c = o + r.normal(0, .3, o.shape); cc = A.advise(c, list(c), "X")
    assert cc["recommended"] is None and cc["allowed"]["knn"].startswith("Duyarlılık")
    e = o.copy(); e["e"] = np.nan; z = A.advise(e, list(e), "X")
    assert set(z["allowed"]) == {"listwise"} and "e" in z["pattern"]["all_missing_items"]
    assert "mekanizması" in a["mechanism_note"] and "kesin olarak belirlenemez" in a["mechanism_note"]


def test_irt_screen_keeps_missing_item_and_shows_advice():
    from streamlit.testing.v1 import AppTest
    x = binary(); x.iloc[[2, 5], 3] = np.nan
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.session_state["raw_df"] = x; at.session_state["scored_df"] = x; at.session_state["item_cols"] = list(x)
    at.session_state["nav_page"] = "MTK: 2PL"; at.run()
    assert not at.exception
    text = " ".join(w.value for w in at.warning)
    assert "Kayıp veri danışmanı" in text and "önerilen birincil yöntem yoktur" in text
    assert not any("dışarıda bırakılacak" in i.value for i in at.info)               # V4 is not silently excluded
    assert any("Kayıp Veri İşlemleri'ne git" in b.label for b in at.button)
    assert at.session_state["scored_df"].isna().sum().sum() == 2                       # nothing imputed or deleted


def test_complete_data_has_no_advisor_and_no_banner():
    from streamlit.testing.v1 import AppTest
    x = binary()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.session_state["raw_df"] = x; at.session_state["scored_df"] = x; at.session_state["item_cols"] = list(x)
    at.session_state["nav_page"] = "MTK: 2PL"; at.run()
    assert not at.exception and not any("Kayıp veri" in w.value for w in list(at.warning) + list(at.info))


def test_handoff_applies_on_missing_page_keeps_original_and_returns():
    from streamlit.testing.v1 import AppTest
    x = binary(); x.iloc[[2, 5], 3] = np.nan
    at = AppTest.from_file(str(ROOT / "pages/90_Kayip_Veri_Analizi_ve_Atama.py"), default_timeout=300)
    at.session_state["raw_df"] = x; at.session_state["scored_df"] = x; at.session_state["item_cols"] = list(x)
    adv = A.advise(x, list(x), "MTK: 2PL")
    at.session_state["missing_advice"] = dict(page="MTK: 2PL", source="scored_df", columns=list(x), recommended=adv["recommended"],
                                              allowed=list(adv["allowed"]), reason=adv["reason"])
    at.run(); assert not at.exception
    radio = at.radio[0]; assert radio.value == "listwise" and len(radio.options) == len(adv["allowed"])   # only advised methods
    next(b for b in at.button if "Seçili yöntemi uygula" in b.label).click().run()
    try:
        next(b for b in at.button if "analizde kullan" in b.label).click().run()
    except Exception:
        pass                                                                          # switch_page to app.py inside AppTest
    s = at.session_state
    assert len(s["scored_df"]) == len(x) - 2 and s["_missing_original_scored_df"].equals(x)   # original kept
    rec = s["missing_applied"]; assert rec["method"] == "listwise" and rec["n_before"] == 400 and rec["n_after"] == 398
    assert s["nav_page"] == "MTK: 2PL"


def test_other_analyses_get_the_shared_advice_without_data_change():
    from streamlit.testing.v1 import AppTest
    x = binary(); x.iloc[[2, 5], 3] = np.nan
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.session_state["raw_df"] = x; at.session_state["scored_df"] = x; at.session_state["item_cols"] = list(x)
    at.session_state["nav_page"] = "4. Test İstatistikleri"; at.run()
    assert not at.exception and any("Kayıp veri danışmanı" in w.value and "%80" in w.value for w in at.warning)
    assert at.session_state["scored_df"].isna().sum().sum() == 2
    assert not at.dataframe and any("onaylamanızı" in i.value for i in at.info)       # analysis waits for approval
    at.checkbox[0].check().run()
    assert not at.exception and at.session_state["missing_rule_approved"] and at.session_state["scored_df"].isna().sum().sum() == 2
