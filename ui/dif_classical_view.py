"""Observed-score DIF page: Mantel–Haenszel / Mantel and (ordinal) logistic regression."""
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.dif_classical import ADJUST_METHODS, dif_screen
from reports.excel_export import sanitize_excel_cell

ADJUST_LABELS = {"BH": "Benjamini–Hochberg (FDR)", "BY": "Benjamini–Yekutieli (FDR, bağımlılığa dayanıklı)",
                 "Holm": "Holm (FWER)", "Bonferroni": "Bonferroni (FWER)", "none": "Düzeltme yok"}


def _safe(frame):
    out = frame.copy()
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]):
            out[c] = out[c].map(sanitize_excel_cell)
    out.columns = [sanitize_excel_cell(str(c)) for c in out.columns]
    return out


def render(data, item_cols=(), id_col=None, group_col=None):
    st.subheader("⚖️ Gözlenen Puan DIF — Mantel–Haenszel ve Lojistik Regresyon")
    st.caption("İkili maddeler: MH (Holland–Thayer; ETS A/B/C) ve lojistik regresyon (tek biçimli / tek biçimli "
               "olmayan; Jodoin–Gierl ΔR²). Sıralı maddeler: Mantel testi + SMD ve sıralı lojistik regresyon. "
               "Eşleştirme: toplam puan (incelenen madde dahil); isteğe bağlı saflaştırma.")
    st.warning("Bu yöntemler MTK olabilirlik DIF'i değildir. Gruplar arasında yetenek farkı (impact) varken ve "
               "madde ayırt edicilikleri farklıyken I. tip hata şişebilir; ayrıntı validation/evidence/p2_dif.")
    available = [c for c in data.columns if c not in (id_col,)]
    gcol = st.selectbox("Grup değişkeni", [c for c in available if c != id_col],
                        index=([c for c in available].index(group_col) if group_col in available else 0), key="cdif_group")
    levels = [v for v in pd.Series(data[gcol]).dropna().unique().tolist()]
    if len(levels) < 2:
        st.error("Grup değişkeninde en az iki düzey gerekli."); return
    c1, c2 = st.columns(2)
    ref = c1.selectbox("Referans grup", levels, key="cdif_ref")
    foc = c2.selectbox("Odak grup", [v for v in levels if v != ref], key="cdif_foc")
    items = st.multiselect("Maddeler", [c for c in available if c != gcol],
                           default=[c for c in item_cols if c in available and c != gcol], key="cdif_items")
    c3, c4, c5 = st.columns(3)
    method = c3.radio("Yöntem", ["MH", "LR"], horizontal=True, key="cdif_method",
                      format_func=lambda m: "Mantel–Haenszel / Mantel" if m == "MH" else "Lojistik regresyon")
    adjust = c4.selectbox("Çoklu karşılaştırma düzeltmesi", ADJUST_METHODS, format_func=ADJUST_LABELS.get, key="cdif_adj")
    alpha = c5.select_slider("α", [.01, .05, .10], .05, key="cdif_alpha")
    purify = st.checkbox("Eşleştirme puanını iteratif saflaştır", True, key="cdif_purify")
    if len(items) < 3:
        st.info("En az üç madde seçin."); return
    x = data.loc[:, items].apply(pd.to_numeric, errors="coerce")
    ordinal = bool(x.max().max() > 1)
    st.caption("Veri türü: " + ("sıralı (0, 1, 2, ...)" if ordinal else "ikili (0/1)"))
    if st.button("▶️ DIF analizini çalıştır", type="primary", key="cdif_run"):
        try:
            with st.spinner("DIF hesaplanıyor…"):
                frame, meta = dif_screen(data.loc[:, items], data[gcol], ref, foc, method=method, ordinal=ordinal,
                                         adjust=adjust, alpha=alpha, purify=purify)
            st.session_state.cdif_result = (frame, meta, tuple(items), gcol, ref, foc)
        except Exception as exc:
            st.session_state.pop("cdif_result", None)
            st.error(f"DIF analizi yapılamadı: {exc}")
    saved = st.session_state.get("cdif_result")
    if not saved or saved[2:] != (tuple(items), gcol, ref, foc):
        return
    frame, meta = saved[0], saved[1]
    st.caption(f"Referans N = {meta['n_reference']}, odak N = {meta['n_focal']} | Düzeltme: {ADJUST_LABELS[meta['adjust']]} "
               f"| Saflaştırma turu: {meta['rounds']}")
    for note in meta["notes"]:
        st.warning(note)
    show = frame.drop(columns=[c for c in ("reason",) if c in frame.columns and frame[c].eq("").all()])
    st.dataframe(show.round(4), hide_index=True, use_container_width=True)
    st.metric("DIF işaretlenen madde", int(frame["DIF (düzeltilmiş p < alpha)"].sum()))
    if method == "MH" and not meta["ordinal"]:
        fig = go.Figure()
        fig.add_scatter(x=frame["Madde"], y=frame["delta"], mode="markers", name="MH D-DIF",
                        error_y=dict(type="data", array=2.35 * 1.96 * frame["se_log_or"]))
        for yv in (-1.5, -1, 1, 1.5):
            fig.add_hline(y=yv, line_dash="dot", line_color="gray")
        fig.update_layout(title="ETS MH D-DIF (−2,35 ln α_MH) ve %95 aralık; negatif = odak grup aleyhine",
                          xaxis_title="Madde", yaxis_title="MH D-DIF", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    if method == "LR":
        fig = go.Figure(); fig.add_bar(x=frame["Madde"], y=frame["delta_r2"], name="Nagelkerke ΔR²")
        fig.add_hline(y=.035, line_dash="dot"); fig.add_hline(y=.07, line_dash="dot")
        fig.update_layout(title="Lojistik regresyon DIF etki büyüklüğü (Jodoin–Gierl sınırları .035 / .070)",
                          xaxis_title="Madde", yaxis_title="ΔR²", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    meta_frame = pd.DataFrame([(k, str(v)) for k, v in meta.items()], columns=["Alan", "Değer"])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        _safe(frame).to_excel(w, sheet_name="DIF", index=False)
        _safe(meta_frame).to_excel(w, sheet_name="Meta", index=False)
    d1, d2 = st.columns(2)
    d1.download_button("DIF tablosu — CSV", _safe(frame).to_csv(index=False).encode("utf-8-sig"),
                       f"dif_{method}.csv", "text/csv", key="cdif_csv")
    d2.download_button("DIF tablosu — XLSX", buf.getvalue(), f"dif_{method}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="cdif_xlsx")
