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


@pytest.mark.parametrize("page", ["MTK: 1PL", "MTK: 2PL", "MTK: 3PL", "MTK: 4PL", "MTK: Karşılaştırma"])
def test_binary_irt_recommendations_are_supported_category_safe_and_deterministic(page):
    one = binary(); one.iloc[3, 3] = np.nan                                  # a single missing V4 response
    a = A.advise(one, list(one), page)
    assert a["status"] == "action" and a["recommended"] == "listwise" and A.engine(page)["native"] is False
    heavy = binary(rate=.05); b = A.advise(heavy, list(heavy), page)
    assert b["recommended"] == "mode" and "median" in b["excluded"] and "keep" in b["excluded"]
    assert set(b["allowed"]) | {b["recommended"]} <= set(METHODS) and "Çoklu atama sunulmaz" in b["multiple_imputation"]
    assert A.advise(heavy, list(heavy), page) == b                           # deterministic
    out = apply(heavy, list(heavy), b["recommended"])
    assert set(pd.unique(out.to_numpy().ravel())) <= {0., 1.}                 # categories preserved
    obs = heavy.notna(); assert ((out == heavy) | ~obs).all().all()           # observed values untouched


def test_ordinal_continuous_and_all_missing_rules():
    r = np.random.default_rng(2); o = pd.DataFrame(r.integers(1, 6, (300, 5)).astype(float), columns=list("abcde")).mask(r.random((300, 5)) < .05)
    a = A.advise(o, list(o), "Bilinmeyen analiz")
    assert a["engine"]["native"] is None and a["recommended"] == "mode" and {"median", "knn", "zero"} <= set(a["excluded"])
    c = o + r.normal(0, .3, o.shape); assert A.advise(c, list(c), "X")["recommended"] == "knn"
    e = o.copy(); e["e"] = np.nan; z = A.advise(e, list(e), "X")
    assert z["recommended"] == "listwise" and "e" in z["pattern"]["all_missing_items"]
    assert "mekanizması" in a["mechanism_note"] and "kesin olarak belirlenemez" in a["mechanism_note"]


def test_irt_screen_keeps_missing_item_and_shows_advice():
    from streamlit.testing.v1 import AppTest
    x = binary(); x.iloc[[2, 5], 3] = np.nan
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.session_state["raw_df"] = x; at.session_state["scored_df"] = x; at.session_state["item_cols"] = list(x)
    at.session_state["nav_page"] = "MTK: 2PL"; at.run()
    assert not at.exception
    text = " ".join(w.value for w in at.warning)
    assert "Kayıp veri danışmanı" in text and "tam kayıt" in text
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
    assert not at.exception and any("Kayıp veri danışmanı" in w.value and "doğrulanmadı" in w.value for w in at.warning)
    assert at.session_state["scored_df"].isna().sum().sum() == 2
