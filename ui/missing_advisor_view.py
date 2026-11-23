"""Shared missing-data advisor panel, hand-off to the Kayıp Veri İşlemleri page and the 'applied method' banner."""
import pandas as pd
import streamlit as st

import core.missing_advisor as A

MISSING_PAGE_FILE = "pages/90_Kayip_Veri_Analizi_ve_Atama.py"
# Routes without item-matrix analyses or with their own long-format handling (MFRM) get no global panel.
SKIP = ("1. Ana Sayfa", "2. Veri Yükleme", "3. Puanlama", "12. Kullanım", "MTK: RaschPy MFRM", "MTK: Çok Yüzeyli Rasch",
        "MTK: 1PL", "MTK: 2PL", "MTK: 3PL", "MTK: 4PL", "MTK: Karşılaştırma")


def source_key():
    return "scored_df" if st.session_state.get("scored_df") is not None else "raw_df"


def _fp(df):
    return (df.shape, int(pd.util.hash_pandas_object(df, index=True).sum())) if df is not None else None


def applied():
    """The applied record if it still matches the active data; a stale record (new data loaded) is dropped."""
    rec = st.session_state.get("missing_applied")
    if rec and _fp(st.session_state.get(rec["source"])) != rec["fingerprint"]:
        st.session_state.pop("missing_applied", None); st.session_state.pop(f"_missing_original_{rec['source']}", None); rec = None
    return rec


def banner():
    rec = applied()
    if not rec:
        return
    st.info(f"Kayıp veri işlemi uygulandı: **{A.method_label(rec['method'])}** ({rec['columns_n']} madde; {rec['cells']} kayıp hücre; "
            f"satır {rec['n_before']} → {rec['n_after']}). Analizler işlenmiş veriyle çalışır; orijinal veri saklıdır. "
            "Tek atama standart hataları küçümseyebilir.")
    if st.button("↩ Orijinal veriye dön", key="missing_revert"):
        st.session_state[rec["source"]] = st.session_state.pop(f"_missing_original_{rec['source']}")
        st.session_state.pop("missing_applied", None); st.rerun()


def panel(page, frame, columns, key="missing_advisor"):
    """Render the advice for `columns` of `frame` on `page`; returns the advice dict (None when nothing is missing)."""
    cols = [c for c in columns if c in frame.columns]
    if not cols or not frame[cols].isna().to_numpy().any():
        return None
    adv = A.advise(frame, cols, page); p = adv["pattern"]
    box = st.success if adv["status"] == "native" else st.warning
    box(f"**Kayıp veri danışmanı:** {p['missing_cells']} kayıp hücre (%{p['missing_pct']:.1f}); {p['rows_with_missing']} satır "
        f"(%{p['row_loss_pct']:.1f}). {adv['reason']}")
    with st.expander("Kayıp veri önerisinin gerekçesi", expanded=False):
        st.caption(adv["mechanism_note"] + " " + adv["multiple_imputation"])
        if adv["allowed"]:
            st.markdown("**Uygun yöntemler:**\n" + "\n".join(f"- {A.method_label(m)}{' — önerilen' if m == adv['recommended'] else ''}: {r}"
                                                             for m, r in adv["allowed"].items()))
        if adv["excluded"]:
            st.markdown("**Önerilmeyen yöntemler:**\n" + "\n".join(f"- {A.method_label(m)}: {r}" for m, r in adv["excluded"].items()))
        st.caption("Kaynaklar: " + "; ".join(adv["references"]))
    if adv["status"] == "action" and st.button(f"🧩 Kayıp Veri İşlemleri'ne git (önerilen: {A.method_label(adv['recommended'])})",
                                               key=f"{key}_go"):
        st.session_state["missing_advice"] = dict(page=page, source=source_key(), columns=cols, recommended=adv["recommended"],
                                                  allowed=list(adv["allowed"]), reason=adv["reason"])
        st.switch_page(MISSING_PAGE_FILE)
    return adv


def global_panel(page):
    if any(str(page).startswith(s) for s in SKIP):
        return
    frame = st.session_state.get(source_key()); cols = list(st.session_state.get("item_cols") or ())
    if frame is not None and cols:
        panel(page, frame, cols, key="missing_advisor_global")


def accept(result, method, n_before, cells):
    """Called on the Kayıp Veri page: keep the original, make the processed frame active, return to the analysis."""
    adv = st.session_state.get("missing_advice"); src = adv["source"]
    st.session_state.setdefault(f"_missing_original_{src}", st.session_state.get(src))
    st.session_state[src] = result
    st.session_state["missing_applied"] = dict(source=src, method=method, page=adv["page"], columns_n=len(adv["columns"]),
                                               cells=cells, n_before=n_before, n_after=len(result), fingerprint=_fp(result))
    st.session_state["nav_page"] = adv["page"]; st.session_state.pop("missing_advice", None)
    st.switch_page("app.py")
