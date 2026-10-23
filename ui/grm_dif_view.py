"""Experimental GRM likelihood DIF, visibly separate from score screens."""
import hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from core import grm_dif as G
from reports.grm_dif_report import report_tables, bundle, safe_frame, CAUTIONS


def result_key(frame, items, group, reference, focal, anchors, categories, points):
    data_hash = pd.util.hash_pandas_object(frame.loc[:, list(items) + [group]], index=True).values.tobytes()
    settings = repr((tuple(items), group, reference, focal, tuple(anchors), categories, points)).encode()
    return hashlib.sha256(data_hash + settings).hexdigest()


def figure(category, score, reference, focal, item):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.12)
    colors = ["#4c78a8", "#e45756", "#59a14f", "#b279a2", "#f28e2b"]
    for c, rows in category.groupby("Kategori"):
        color = colors[int(c) % len(colors)]
        for label, col, dash in ((reference, "P (referans)", "solid"), (focal, "P (odak)", "dash")):
            fig.add_trace(go.Scatter(x=rows.Theta, y=rows[col], mode="lines", name=f"{label}, kategori {c}",
                                     line=dict(color=color, dash=dash)), row=1, col=1)
    fig.add_trace(go.Scatter(x=score.Theta, y=score["%95 GA üst"], mode="lines", line=dict(width=0), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=score.Theta, y=score["%95 GA alt"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(90,90,90,.15)", name="Noktasal %95 GA"), row=2, col=1)
    fig.add_trace(go.Scatter(x=score.Theta, y=score["Beklenen puan farkı"], mode="lines", name="Odak − referans",
                             line=dict(color="black")), row=2, col=1)
    fig.add_hline(y=0, row=2, col=1, line_color="gray")
    fig.update_yaxes(title="Kategori olasılığı", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title="Beklenen puan farkı", row=2, col=1)
    fig.update_xaxes(title="Theta (referans N(0,1))", row=2, col=1)
    fig.update_layout(title=f"{item}: ortak ölçekte GRM", template="plotly_white", height=650)
    return fig


def render(data, item_cols=(), id_col=None, group_col=None):
    st.subheader("GRM: sabit çapalı formel DIF — DENEYSEL")
    st.warning("Bu bölüm model temelli eşik/eğim değişmezliğini sınar. Gözlenen puan ön taraması değildir. "
               "Doğrulama ortak kategorili, sabit çapalı iki grup tasarımıyla sınırlıdır; R mirt karşılaştırması yoktur.")
    with st.expander("Yöntem ve bilimsel kapsam"):
        for note in CAUTIONS: st.write(note)
    group_options = [c for c in data.columns if c != id_col]
    if not group_options:
        st.info("Kimlik dışı grup sütunu gereklidir."); return
    group = st.selectbox("Grup sütunu", group_options,
                        index=group_options.index(group_col) if group_col in group_options else 0, key="grm_group")
    available = [c for c in data.columns if c not in (id_col, group)]
    chosen = st.multiselect("Sıralı puan maddeleri (0,1,...)", available,
                           default=[c for c in item_cols if c in available], key="grm_items")
    levels = data[group].dropna().unique().tolist()
    if len(levels) < 2:
        st.info("En az iki farklı grup gereklidir."); return
    reference = st.selectbox("Referans grup", levels, key="grm_reference")
    focal = st.selectbox("Odak grup", [v for v in levels if v != reference], key="grm_focal")
    anchors = st.multiselect("Bağımsız kanıtla belirlenmiş sabit çapalar", chosen, key="grm_anchors")
    categories = st.selectbox("Tüm maddelerde ortak kategori sayısı", list(range(3, 11)), key="grm_categories")
    points = st.selectbox("Quadrature noktası", [61, 81], key="grm_points")
    agreed = st.checkbox("Yanıtlar gerekçeli sıralı puanlardır; sabit çapa varsayımını ve deneysel kapsamı kabul ediyorum.",
                        key="grm_ack")
    key = result_key(data, chosen, group, reference, focal, anchors, categories, points)
    if st.button("GRM formel DIF hesapla", key="grm_run",
                 disabled=not agreed or len(chosen) < 3 or not 2 <= len(anchors) < len(chosen)):
        try:
            with st.spinner("GRM ortak çok gruplu modelleri kestiriliyor"):
                out = G.test_dif(data.loc[:, chosen], data[group], reference, focal, anchors,
                                 categories=categories, points=points)
            st.session_state["grm_dif_result"] = dict(key=key, out=out)
        except (ValueError, np.linalg.LinAlgError) as exc:
            st.session_state.pop("grm_dif_result", None)
            st.error(str(exc))
    entry = st.session_state.get("grm_dif_result")
    if entry is None: return
    if entry["key"] != key or not agreed:
        st.info("Veri, madde, grup veya çapa ayarı değişti; eski sonuç gösterilmiyor. Yeniden hesaplayın."); return
    out = entry["out"]
    if "tables" not in entry: entry["tables"] = report_tables(out)
    tables = entry["tables"]
    st.caption(out["status"])
    st.dataframe(tables["Yontem"].astype(str), hide_index=True)
    for note in out["warnings"]: st.warning(note)
    st.dataframe(out["table"], hide_index=True)
    st.dataframe(out["parameters"], hide_index=True)
    with st.expander("Yakınsama, bilgi matrisi ve quadrature denetimi"):
        st.dataframe(tables["Optimizasyon"], hide_index=True)
    with st.expander("Betimsel kategori marjin uyumu (formel uygunluk testi değil)"):
        st.dataframe(tables["Kategori_tanilari"], hide_index=True)
    valid_items = out["table"].loc[out["table"].p.notna() & ~out["table"]["Sabit çapa"], "Madde"].tolist()
    if valid_items:
        item = st.selectbox("Koşullu kategori eğrileri ve puan farkı", valid_items, key="grm_plot_item")
        st.plotly_chart(figure(tables["Kategori_egrileri"].loc[lambda f: f.Madde == item],
                              tables["Puan_farki"].loc[lambda f: f.Madde == item], reference, focal, item),
                        use_container_width=True)
    else:
        st.error("Geçerli çıkarım üretilmedi. Başarısızlık notlarını inceleyin; bu sonuç DIF yokluğu kanıtı değildir.")
    st.download_button("Madde tablosu CSV (UTF-8 BOM)", safe_frame(out["table"]).to_csv(index=False).encode("utf-8-sig"),
                       "GRM_DIF_maddeler.csv", "text/csv", key="grm_csv")
    if st.button("DOCX / XLSX / CSV / 300 dpi PNG paketini hazırla", key="grm_prepare"):
        entry["bundle"] = bundle(out, tables)
    if "bundle" in entry:
        st.download_button("GRM araştırma paketini ZIP indir", entry["bundle"], "GRM_DIF.zip", "application/zip", key="grm_zip")
