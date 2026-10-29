"""Opt-in PCM/GPCM/RSM/NRM DIF, separate from existing score screens."""
import hashlib
import json
import numpy as np
import pandas as pd
import streamlit as st
from core.reproducible import displayed
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core import polytomous_dif as P
from reports.polytomous_dif_report import report_tables, bundle, safe_frame, COMMON_CAUTION, MODEL_CAUTION


def result_key(data, items, group, reference, focal, anchors, model, categories, labels, points):
    rows = pd.util.hash_pandas_object(data.loc[:, list(items) + [group]], index=True).values.tobytes()
    settings = repr((tuple(items), group, reference, focal, tuple(anchors), model, categories, labels, points)).encode()
    return hashlib.sha256(rows + settings).hexdigest()


def figure(out, tables, item):
    nominal = out["model"] == "NRM"
    fig = make_subplots(rows=2 if nominal else 3, cols=1, shared_xaxes=True)
    for j, (c, rows) in enumerate(tables["Kategori_egrileri"].loc[lambda f: f.Madde == item].groupby("Kategori", sort=False)):
        color = ["#4c78a8", "#e45756", "#59a14f", "#b279a2", "#f28e2b"][j % 5]
        for group, col, dash in ((out["reference"], "P (referans)", "solid"), (out["focal"], "P (odak)", "dash")):
            fig.add_trace(go.Scatter(x=rows.Theta, y=rows[col], name=f"{group}: {c}", line=dict(color=color, dash=dash)), row=1, col=1)
        fig.add_trace(go.Scatter(x=rows.Theta, y=rows["Olasılık farkı"], name=f"ΔP: {c}", line=dict(color=color)), row=2, col=1)
    if not nominal:
        score = tables["Puan_farki"].loc[lambda f: f.Madde == item]
        fig.add_trace(go.Scatter(x=score.Theta, y=score["Beklenen puan farkı"], name="Δ beklenen puan"), row=3, col=1)
    fig.update_xaxes(title_text="Theta (referans N(0,1))", row=2 if nominal else 3, col=1)
    fig.update_yaxes(title_text="Kategori olasılığı", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Odak − referans olasılık", row=2, col=1)
    fig.update_layout(title=f"{item}: {out['model']} (noktasal aralıklar indirilebilir tabloda/PNG'de)", height=750)
    return fig


def render(data, item_cols=(), id_col=None, group_col=None):
    st.subheader("PCM / GPCM / RSM / NRM: sabit çapalı formel DIF — DENEYSEL")
    st.warning(COMMON_CAUTION)
    model = st.selectbox("Formel model", P.MODELS, key="poly_model")
    st.warning(MODEL_CAUTION[model])
    st.caption({"PCM": "Sabit a=1; sıralanması gerekmeyen C−1 adımın değişmezliği (C−1 sd).",
        "GPCM": "Pozitif eğim ve serbest adımlar; genel C sd, koşullu adım C−1 sd, koşullu eğim 1 sd.",
        "RSM": "Tüm maddeler/gruplarda ortak, toplamı sıfır kategori adımları; yalnız madde konumu DIF'i (1 sd).",
        "NRM": "Sırasız etiketler, referans kategori; genel 2(C−1) sd, koşullu kesişim/eğim C−1 sd. Puan/başarı sıralaması yok."}[model])
    options = [c for c in data if c != id_col]
    if not options: st.info("Kimlik dışı grup sütunu gereklidir."); return
    group = st.selectbox("Grup sütunu", options, index=options.index(group_col) if group_col in options else 0, key="poly_group")
    available = [c for c in data if c not in (id_col, group)]
    items = st.multiselect("Maddeler (kimlik/grup hariç)", available, default=[c for c in item_cols if c in available], key="poly_items")
    groups = data[group].dropna().unique().tolist()
    if len(groups) < 2: st.info("İki farklı grup gereklidir."); return
    reference = st.selectbox("Referans grup", groups, key="poly_reference")
    focal = st.selectbox("Odak grup", [g for g in groups if g != reference], key="poly_focal")
    anchors = st.multiselect("Bağımsız kanıtla önceden belirlenmiş sabit çapalar", items, key="poly_anchors")
    categories = st.selectbox("Ortak kategori sayısı", list(range(3, 11)), key="poly_categories")
    labels, label_error = None, None
    if model == "NRM":
        text = st.text_input("Kategori etiketleri JSON listesi; ilk etiket referans, sıra başarı sırası değildir", "[0, 1, 2]", key="poly_labels")
        try:
            labels = json.loads(text)
            if not isinstance(labels, list) or len(labels) != categories or any(not isinstance(v, (str, int, float)) for v in labels): raise ValueError()
        except (ValueError, TypeError): label_error = "Kategori etiketlerini kategori sayısına eşit bir JSON listesi olarak belirtin."
        if label_error: st.error(label_error)
        st.info("Yön tanımı: seçilen ilk çapanın ikinci/referans kategori eğim farkı pozitif kabul edilir; bu bir başarı yönü değildir.")
    points = st.selectbox("Quadrature noktası", [61, 81], key="poly_points")
    agreed = st.checkbox("Modelin puan/kategori anlamını, bağımsız sabit çapaları ve deneysel doğrulama sınırlarını kabul ediyorum.", key="poly_ack")
    key = result_key(data, items, group, reference, focal, anchors, model, categories, labels, points)
    if st.button("Formel DIF hesapla", key="poly_run", disabled=not agreed or label_error is not None or len(items) < 3 or not 2 <= len(anchors) < len(items)):
        try:
            with st.spinner(f"{model} iç içe modelleri kestiriliyor"):
                out = P.test_dif(data.loc[:, items], data[group], reference, focal, anchors, model=model,
                    categories=categories, category_labels=labels, points=points)
            st.session_state["poly_dif_result"] = dict(key=key, out=out)
        except (ValueError, np.linalg.LinAlgError) as exc:
            st.session_state.pop("poly_dif_result", None); st.error(str(exc))
    entry = st.session_state.get("poly_dif_result")
    if entry is None: return
    if entry["key"] != key or not agreed or label_error:
        st.info("Veri/model/ayar değişti; eski sonuç gösterilmiyor. Yeniden hesaplayın."); return
    out = entry["out"]
    displayed(out)
    if "tables" not in entry: entry["tables"] = report_tables(out)
    tables = entry["tables"]
    st.caption(out["status"])
    for note in out["warnings"]: st.warning(note)
    st.dataframe(tables["Yontem"].astype(str), hide_index=True)
    for name in ("Maddeler", "Parametreler", "Etkiler"): st.dataframe(tables[name], hide_index=True)
    with st.expander("Optimizasyon / betimsel kategori tanıları"):
        st.dataframe(tables["Optimizasyon"], hide_index=True); st.dataframe(tables["Kategori_tanilari"], hide_index=True)
    valid = out["table"].loc[lambda f: f.p.notna() & ~f["Sabit çapa"], "Madde"].tolist()
    if valid:
        item = st.selectbox("Ortak theta'da kategori karşılaştırması", valid, key="poly_plot")
        st.plotly_chart(figure(out, tables, item), use_container_width=True)
    else: st.error("Geçerli çıkarım yok; başarısız kalibrasyon DIF yokluğu değildir.")
    st.download_button("Madde CSV (UTF-8 BOM)", safe_frame(out["table"]).to_csv(index=False).encode("utf-8-sig"),
                       f"{model}_DIF_maddeler.csv", "text/csv", key="poly_csv")
    if st.button("DOCX / XLSX / CSV / 300 dpi PNG paketini hazırla", key="poly_prepare"): entry["bundle"] = bundle(out, tables)
    if "bundle" in entry: st.download_button("Araştırma ZIP paketini indir", entry["bundle"], f"{model}_DIF.zip", "application/zip", key="poly_zip")
