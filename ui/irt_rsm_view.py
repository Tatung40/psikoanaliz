"""Rating Scale Model page: complete single-group RSM analysis (native MML, core.irt_rsm).

Separate from the fixed-anchor RSM DIF page (core.polytomous_dif) and never labeled PCM.
"""
import hashlib
import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import irt_rsm as R
from reports.excel_export import sanitize_excel_cell

TABS = ["⚙️ Model ve Ayarlar", "📋 Genel Sonuçlar", "🧩 Madde ve Adım Parametreleri", "👤 Kişi Yetenekleri",
        "📈 Grafikler", "🔎 Model Uyumu", "📦 İndirme"]


def _fingerprint(frame):
    h = hashlib.sha256(repr((tuple(map(str, frame.columns)), frame.shape)).encode())
    h.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return h.hexdigest()


@st.cache_data(show_spinner=False, max_entries=6)
def _analyse(frame, identification, points, with_lr):
    f = R.fit(frame, "RSM", identification, points)
    out = {"fit": {k: v for k, v in f.items() if k not in ("spec",)}, "persons": R.persons(f),
           "item_fit": R.item_fit(f)}
    out["fit"]["spec_model"] = "RSM"
    if with_lr:
        lr = R.rsm_vs_pcm(frame, identification, points)
        out["lr"] = {"LR": lr["LR"], "df": lr["df"], "p": lr["p"], "valid": lr["valid"], "note": lr["note"],
                     "table": lr["table"]}
    return out


def _safe(frame):
    out = frame.copy()
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]):
            out[c] = out[c].map(sanitize_excel_cell)
    out.columns = [sanitize_excel_cell(str(c)) for c in out.columns]
    return out


def metadata(f, identification, points, n_rows):
    import importlib.metadata as md
    return pd.DataFrame([
        ("Model", "RSM (Andrich derecelendirme ölçeği modeli) — PCM değildir"),
        ("Motor", "EserPsychometrics yerel MML (core.irt_rsm)"),
        ("Tanımlama", "θ ~ N(0, σ²), σ serbest; Στ = 0" if identification == "rasch" else "θ ~ N(0, 1); Στ = 0"),
        ("Gauss–Hermite düğümü", points), ("N (analiz)", f["n"]), ("N (yüklenen)", n_rows), ("k", f["k"]),
        ("Kategori sayısı", f["C"]), ("Eksik yanıt (MAR, atlandı)", f["missing"]),
        ("Yakınsama", "Evet" if f["converged"] else "Hayır"), ("Skor normu / kişi", f["max_score_per_person"]),
        ("SH notu", f["se_reason"] or "Gözlenen bilgi matrisi (merkezi fark, analitik skor)"),
        ("numpy", md.version("numpy")), ("scipy", md.version("scipy")),
        ("Tarih", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")),
        ("Doğrulama", "validation/evidence/p3_rsm: Python mirt EMEstimator(RatingScaleModel) ile logLik/parametre "
                      "karşılaştırması ve önceden tanımlı kurtarma simülasyonu"),
    ], columns=["Alan", "Değer"]).astype({"Değer": str})


def render(data, item_cols=(), id_col=None, group_col=None):
    st.subheader("📏 Derecelendirme Ölçeği Modeli (RSM) — tam analiz")
    st.caption("Andrich (1978) RSM: tüm maddeler ortak adım (eşik) parametrelerini paylaşır; maddeler yalnızca "
               "konum (δ) bakımından ayrışır. Bu sayfa PCM değildir; PCM yalnızca iç içe model karşılaştırmasında "
               "alternatif olarak kestirilir. Çok kategorili DIF sayfasındaki sabit-çapalı RSM DIF'ten ayrıdır.")
    available = [c for c in data.columns if c not in (id_col, group_col)]
    tabs = st.tabs(TABS)
    with tabs[0]:
        cols = st.multiselect("Madde sütunları (0, 1, 2, ... sıralı puanlar)", available,
                              default=[c for c in item_cols if c in available] or available[:10], key="rsm_items")
        ident = st.radio("Örtük ölçek tanımlaması", ["rasch", "unit_variance"], horizontal=True, key="rsm_ident",
                         format_func=lambda v: "Rasch: σ serbest (önerilen)" if v == "rasch" else "σ = 1 sabit")
        points = st.select_slider("Gauss–Hermite düğüm sayısı", [21, 41, 61], 41, key="rsm_points")
        with_lr = st.checkbox("RSM ile PCM'yi iç içe olabilirlik oranı testiyle karşılaştır", True, key="rsm_lr")
        if st.button("▶️ RSM analizini başlat", type="primary", key="rsm_run"):
            try:
                if len(cols) < 2:
                    raise ValueError("En az iki madde seçin.")
                frame = data.loc[:, cols]
                with st.spinner("RSM kestiriliyor…"):
                    _analyse(frame, ident, points, with_lr)
                st.session_state.rsm_request = {"cols": list(cols), "ident": ident, "points": points,
                                                "lr": with_lr, "fp": _fingerprint(frame)}
            except Exception as exc:
                st.session_state.pop("rsm_request", None)
                st.error(f"RSM analizi yapılamadı: {exc}")
    req = st.session_state.get("rsm_request")
    res = None
    if req is not None:
        try:
            frame = data.loc[:, req["cols"]]
            if _fingerprint(frame) != req["fp"]:
                raise KeyError
            res = _analyse(frame, req["ident"], req["points"], req["lr"])
        except Exception:
            st.session_state.pop("rsm_request", None)
            with tabs[0]:
                st.info("Veri veya madde seçimi değişti; analizi yeniden başlatın.")
            res = None
    if res is None:
        for t in tabs[1:]:
            with t:
                st.info("Önce **Model ve Ayarlar** sekmesinden analizi başlatın.")
        return
    f, pers, ifit = res["fit"], res["persons"], res["item_fit"]
    tables = f["tables"]
    with tabs[1]:
        c = st.columns(5)
        c[0].metric("N", f["n"]); c[1].metric("Madde", f["k"]); c[2].metric("Kategori", f["C"])
        c[3].metric("AIC", f"{f['aic']:,.2f}"); c[4].metric("BIC", f"{f['bic']:,.2f}")
        st.dataframe(pd.DataFrame([{"logLik": f["loglik"], "Parametre": f["npar"], "Yineleme": f["iterations"],
                                    "Yakınsama": "Evet" if f["converged"] else "Hayır",
                                    "Skor/kişi": f["max_score_per_person"]}]), hide_index=True, use_container_width=True)
        st.dataframe(tables["latent"], hide_index=True, use_container_width=True)
        if not f["converged"]:
            st.error("Kestirim yakınsamadı veya sınırda: SH ve güven aralıkları raporlanmadı.")
        if f["missing"]:
            st.warning(f"{f['missing']} eksik yanıt olabilirlikten çıkarıldı (MAR varsayımı).")
        if res.get("lr"):
            lr = res["lr"]
            st.markdown("**RSM – PCM iç içe model karşılaştırması**")
            st.dataframe(lr["table"], hide_index=True, use_container_width=True)
            if lr["valid"]:
                st.caption(f"LR = {lr['LR']:.3f}, sd = {lr['df']}, p = {lr['p']:.4g}. Anlamlı sonuç ortak adım "
                           "varsayımının veriyle çeliştiğini gösterir (PCM daha uygun olabilir).")
            else:
                st.warning(lr["note"])
    with tabs[2]:
        st.markdown("**Madde konumları (δ)**")
        st.dataframe(tables["items"].round(4), hide_index=True, use_container_width=True)
        st.markdown("**Ortak adım parametreleri (τ, Στ = 0)**")
        st.dataframe(tables["steps"].round(4), hide_index=True, use_container_width=True)
        if np.any(np.diff(f["tau"]) < 0):
            st.warning("Adım parametreleri sıralı değil (τ_s+1 < τ_s). Bu bir kestirim hatası değildir; bazı "
                       "kategorilerin hiçbir θ düzeyinde en olası kategori olmadığını gösterir.")
        st.caption(f["se_reason"] or "SH: gözlenen bilgi matrisinin tersi; %95 Wald aralıkları.")
    with tabs[3]:
        st.dataframe(pers.head(500).round(4), hide_index=True, use_container_width=True)
        st.caption(f"İlk 500 satır ({len(pers)} kişi). Kişi kimlikleri gizlilik nedeniyle tabloya eklenmez. "
                   "WLE uç puanlarda da sonludur; EAP önsel N(0, σ²) kullanır.")
    with tabs[4]:
        theta, P = R.category_curves({"steps": f["steps"]})
        item = st.selectbox("Madde", f["names"], key="rsm_curve_item")
        j = f["names"].index(item)
        fig = go.Figure()
        for cat in range(f["C"]):
            fig.add_scatter(x=theta, y=P[:, j, cat], mode="lines", name=f"Kategori {cat}")
        fig.update_layout(title=f"Kategori Tepki Eğrileri — {item} (RSM)", xaxis_title="θ (logit)",
                          yaxis_title="P(X = c | θ)", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        info = R.information(theta, f["steps"]); tif = info.sum(1)
        fig2 = go.Figure()
        fig2.add_scatter(x=theta, y=info[:, j], name=f"Madde bilgisi — {item}")
        fig2.add_scatter(x=theta, y=tif, name="Test bilgisi", yaxis="y2")
        fig2.update_layout(title="Madde ve Test Bilgi Fonksiyonu (I = Var(X|θ))", xaxis_title="θ",
                           yaxis_title="Madde bilgisi", yaxis2=dict(title="Test bilgisi", overlaying="y", side="right"),
                           template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)
        fig3 = go.Figure(); fig3.add_scatter(x=theta, y=1 / np.sqrt(tif), name="SEM(θ)")
        fig3.update_layout(title="Koşullu Ölçme Standart Hatası", xaxis_title="θ", yaxis_title="SEM",
                           template="plotly_white")
        st.plotly_chart(fig3, use_container_width=True)
        fig4 = go.Figure()
        fig4.add_histogram(y=pers["WLE θ"], name="Kişiler (WLE)", nbinsy=30, opacity=.7)
        for name, d in zip(f["names"], f["delta"]):
            fig4.add_scatter(x=[0], y=[d], mode="markers+text", text=[name], textposition="middle left",
                             marker=dict(symbol="diamond", size=9), showlegend=False)
        fig4.update_layout(title="Kişi–Madde Haritası (WLE θ ve madde konumları δ)", yaxis_title="Logit",
                           xaxis_title="Kişi sayısı", template="plotly_white")
        st.plotly_chart(fig4, use_container_width=True)
    with tabs[5]:
        st.dataframe(ifit.round(3), hide_index=True, use_container_width=True)
        st.caption("Infit/Outfit MNSQ ve ZSTD: WLE θ ile, uç puanlı kişiler hariç. θ aynı veriden kestirildiği "
                   "için kısa testlerde MNSQ beklenen değeri 1'in biraz altındadır; ZSTD'yi bu nedenle temkinle "
                   "yorumlayın. Bilgi fonksiyonu Rasch ailesinde koşullu varyansa eşittir.")
    with tabs[6]:
        meta = metadata(f, req["ident"], req["points"], len(data))
        sheets = {"Meta": meta, "Madde konumları": tables["items"], "Ortak adımlar": tables["steps"],
                  "Örtük dağılım": tables["latent"], "Madde uyumu": ifit, "Kişiler": pers}
        if res.get("lr"):
            sheets["RSM-PCM LR"] = res["lr"]["table"].assign(LR=res["lr"]["LR"], sd=res["lr"]["df"], p=res["lr"]["p"])
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            for name, frame in sheets.items():
                _safe(frame).to_excel(w, sheet_name=name[:31], index=False)
        st.download_button("XLSX — tüm RSM tabloları", buf.getvalue(), "rsm_tum_tablolar.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="rsm_xlsx")
        c1, c2 = st.columns(2)
        c1.download_button("CSV — madde konumları", _safe(tables["items"]).to_csv(index=False).encode("utf-8-sig"),
                           "rsm_madde_konumlari.csv", "text/csv", key="rsm_csv_items")
        c2.download_button("CSV — kişi yetenekleri", _safe(pers).to_csv(index=False).encode("utf-8-sig"),
                           "rsm_kisiler.csv", "text/csv", key="rsm_csv_persons")
