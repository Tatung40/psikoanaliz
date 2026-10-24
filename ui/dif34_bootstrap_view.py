"""3PL/4PL DIF for one studied item: LR statistic with a parametric-bootstrap reference distribution.

The size study is limited evidence, not a model-wide inference permission.
Exact eligibility cannot be established from observed data; p-values stay disabled.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.irt_dif34_bootstrap import bootstrap_dif_test
from reports.excel_export import sanitize_excel_cell

EVIDENCE = Path(__file__).resolve().parents[1] / "validation" / "evidence" / "p2_dif34_bootstrap" / "dif34_bootstrap_size_study.json"


def _enabled():
    try:
        data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        return data["ui_enabled"], {r["condition"]: r for r in data["summary"]}
    except (OSError, ValueError, KeyError):
        return {"3PL": False, "4PL": False}, {}


def result_key(data, items, group, reference, focal, studied, anchors, model, B):
    """Track values (not just shape), row order/index, schema and every fit setting."""
    frame = pd.DataFrame(data)
    values = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    settings = repr((tuple(frame.columns), tuple(map(str, frame.dtypes)), frame.shape,
                     tuple(items), group, reference, focal, studied, tuple(anchors), model,
                     int(B), 2026, 41, .10, "bootstrap-ui-v2")).encode()
    return hashlib.sha256(values + settings).hexdigest()


def inference_eligibility(model):
    if model == "4PL":
        return False, ("4PL için sınırlı warp-speed benzetim kanıtı, bu veri/çapa/parametre tasarımının "
                       "ve sonlu bootstrap testinin uygunluğunu belirlemeye yetmez; p değeri kapalıdır.")
    return False, "3PL büyüklük çalışmasında ölçüt karşılanmadı (≤ .075); p değeri kapalıdır."


def result_table(res, model):
    eligible, reason = inference_eligibility(model)
    show_p = eligible and np.isfinite(res["p"])
    return pd.DataFrame([{"Madde": res["item"], "Model": model, "LR (gözlenen)": res["T_obs"],
        "Bootstrap p": res["p"] if show_p else np.nan,
        "Geçerli tekrar": res.get("B_ok", 0), "Başarısız tekrar": res.get("B_failed", 0),
        "Sınırda parametre (H0/H1)": f"{res['boundary_null']}/{res['boundary_alt']}",
        "Durum": res["status"] + ("" if show_p else " — " + reason)}])


def render(data, item_cols=(), id_col=None, group_col=None):
    st.subheader("⚖️ 3PL/4PL DIF — parametrik bootstrap LR (tek madde, önceden belirlenmiş çapalar)")
    st.caption("3PL/4PL'de c (veya d) sıklıkla sınırda kestirilir; bu durumda χ² referansı geçersizdir ve "
               "asimptotik p değeri verilmez. Bu sayfa LR istatistiğinin sıfır dağılımını, kestirilen sıfır "
               "modelinden (incelenen madde gruplar arasında eşit, odak ortalama/SS serbest) simülasyonla elde eder.")
    candidates = [c for c in data.columns if c != id_col]
    if not candidates:
        st.session_state.pop("b34_result", None)
        st.error("Grup ve madde sütunları gerekli."); return
    gcol = st.selectbox("Grup değişkeni", candidates,
                        index=candidates.index(group_col) if group_col in candidates else 0, key="b34_group")
    levels = pd.Series(data[gcol]).dropna().unique().tolist()
    if len(levels) < 2:
        st.session_state.pop("b34_result", None)
        st.error("Grup değişkeninde en az iki düzey gerekli."); return
    c1, c2, c3 = st.columns(3)
    ref = c1.selectbox("Referans", levels, key="b34_ref")
    foc = c2.selectbox("Odak", [v for v in levels if v != ref], key="b34_foc")
    model = c3.radio("Model", ["3PL", "4PL"], horizontal=True, key="b34_model")
    items = [c for c in item_cols if c in data.columns and c != gcol] or [c for c in candidates if c != gcol]
    if not items:
        st.session_state.pop("b34_result", None)
        st.error("En az bir madde seçilmelidir."); return
    studied = st.selectbox("İncelenen madde", items, key="b34_item")
    anchors = st.multiselect("Önceden belirlenmiş çapalar (değişmezlikleri bu analizle test edilmez)",
                             [c for c in items if c != studied], default=[c for c in items if c != studied],
                             key="b34_anchors")
    B = st.select_slider("Bootstrap tekrar sayısı", [99, 199, 499], 199, key="b34_B")
    _, reason = inference_eligibility(model)
    st.warning(reason + " LR istatistiği ve bootstrap dağılımı yalnız betimsel verilir.")
    key = result_key(data, items, gcol, ref, foc, studied, anchors, model, B)
    saved = st.session_state.get("b34_result")
    if saved is not None and (not isinstance(saved, dict) or saved.get("key") != key):
        st.session_state.pop("b34_result", None)
        st.info("Veri veya ayarlar değişti; eski bootstrap sonucu geçersizleştirildi.")
    if st.button("▶️ Bootstrap DIF testini çalıştır", type="primary", key="b34_run"):
        try:
            with st.spinner(f"{B} bootstrap tekrarı; her tekrarda iki model kestirilir…"):
                res = bootstrap_dif_test(data[items], data[gcol], ref, foc, studied, anchors, model=model, B=int(B))
            st.session_state.b34_result = dict(key=key, result=res, model=model)
        except Exception as exc:
            st.session_state.pop("b34_result", None)
            st.error(f"Analiz yapılamadı: {exc}")
    saved = st.session_state.get("b34_result")
    if not saved:
        return
    res, m = saved["result"], saved["model"]
    table = result_table(res, m)
    st.dataframe(table, hide_index=True, use_container_width=True)
    if res.get("null_stats"):
        fig = go.Figure(); fig.add_histogram(x=res["null_stats"], nbinsx=30, name="Bootstrap LR*")
        fig.add_vline(x=res["T_obs"], line_color="red")
        fig.update_layout(title="Sıfır modeli altında bootstrap LR dağılımı (kırmızı: gözlenen)", xaxis_title="LR",
                          yaxis_title="Tekrar", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    out = table.copy()
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]):
            out[c] = out[c].map(sanitize_excel_cell)
    st.download_button("Sonuç — CSV", out.to_csv(index=False).encode("utf-8-sig"), "dif34_bootstrap.csv",
                       "text/csv", key="b34_csv")
