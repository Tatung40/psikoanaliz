"""İki ekran: 'Boyutluluk ve Faktör Analizi' ve 'Yerel Bağımsızlık ve Madde Uyumu'.

İndirilebilir çıktılar: CSV, XLSX, PNG (matplotlib) ve hepsini içeren ZIP.
"""
import io
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import dimensionality as dm
from core import local_fit as lf
from reports.excel_export import sanitize_excel_cell

PAGE_DIM = "Boyutluluk ve Faktör Analizi"
PAGE_LOCAL = "Yerel Bağımsızlık ve Madde Uyumu"


def _safe(df):
    return df.map(sanitize_excel_cell) if hasattr(df, "map") else df


def _xlsx(tables):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in tables.items():
            _safe(df).to_excel(w, sheet_name=name[:31], index=df.index.name is not None or not isinstance(df.index, pd.RangeIndex))
    return buf.getvalue()


def _png_heatmap(df, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(df.to_numpy(float), cmap="RdBu_r", vmin=-1 if df.min().min() < 0 else 0, vmax=1)
    ax.set_xticks(range(len(df.columns)), df.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(df.index)), df.index, fontsize=7)
    ax.set_title(title); fig.colorbar(im, ax=ax); fig.tight_layout()
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=150); plt.close(fig)
    return b.getvalue()


def _png_lines(x, series, title, xlabel, ylabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, y in series.items():
        ax.plot(x, y, marker="o", label=name)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend(); fig.tight_layout()
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=150); plt.close(fig)
    return b.getvalue()


def _zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def _downloads(prefix, tables, pngs, method_text):
    files = {f"{k}.csv": _safe(v).to_csv(index=True).encode("utf-8-sig") for k, v in tables.items()}
    files[f"{prefix}.xlsx"] = _xlsx(tables)
    files.update({f"{k}.png": v for k, v in pngs.items()})
    files["YONTEM_VE_SINIRLILIKLAR.txt"] = method_text.encode("utf-8")
    c = st.columns(3)
    c[0].download_button("XLSX", files[f"{prefix}.xlsx"], f"{prefix}.xlsx", key=f"{prefix}_xlsx")
    c[1].download_button("ZIP (CSV+XLSX+PNG)", _zip(files), f"{prefix}.zip", key=f"{prefix}_zip")
    sel = c[2].selectbox("CSV", list(tables), key=f"{prefix}_csv_sel")
    c[2].download_button("CSV indir", files[f"{sel}.csv"], f"{sel}.csv", key=f"{prefix}_csv")
    for k, v in pngs.items():
        st.download_button(f"PNG: {k}", v, f"{k}.png", key=f"{prefix}_png_{k}")


def _heat(df, title, zmin=-1, zmax=1):
    fig = go.Figure(go.Heatmap(z=df.to_numpy(float), x=list(df.columns), y=list(df.index),
                               colorscale="RdBu", reversescale=True, zmin=zmin, zmax=zmax))
    fig.update_layout(title=title, height=520)
    return fig


DIM_METHOD = """YÖNTEM: Korelasyon türü veri türüne göre seçilir (0/1 → tetrakorik, 3-10 kategori → polikorik,
sürekli → Pearson). Polikorik/tetrakorik: iki aşamalı ML; boş hücrelerde 0.5 düzeltmesi; pozitif tanımlı
olmayan matris özdeğer kırpma ile düzeltilir ve raporlanır. KMO ve Bartlett seçilen matris üzerinden
hesaplanır (polikorik matriste Bartlett yaklaşıktır). Paralel analiz: sütun permütasyonu, aynı korelasyon
türü, %95 nicelik. AFA: temel eksen faktörleştirme (PAF); döndürme varimax/promax.
SINIRLILIKLAR: Tek boyutluluk özeti betimseldir; evrensel kesme puanı kullanılmaz. Heywood durumu
(h² ≥ 1) uyarı olarak gösterilir. Eksik veri: ikili-çiftli (pairwise) korelasyon."""

LOCAL_METHOD = """YÖNTEM: Yen-Q3 (WLE/EAP kişi kestirimleri ile artık korelasyonu), merkezlenmiş Q3 (Q3 − ortalama Q3),
tek faktörlü PAF artık korelasyonları. Madde uyumu: Orlando-Thissen S-X² (mirt uyumlu); BH-FDR.
SERTİFİKA: 1PL/2PL S-X² p ve FDR kararları CERTIFIED (sabit parametre altın testleri + simülasyon).
3PL/4PL: istatistik gösterilir, p/FDR EXPERIMENTAL / NOT CERTIFIED; karar üretilmez.
Çok kategorili modeller: S-X² uygulanmamıştır, ekran kapatılır.
SINIRLILIKLAR: Q3 için evrensel kesme puanı uydurulmaz; isteğe bağlı parametrik bootstrap veri temelli kritik
değer verir. S-X², Pearson X² ve M2 farklı istatistiklerdir; burada yalnızca S-X² raporlanır.
S-X² tam (eksiksiz) veri gerektirir."""


def render_dimensionality(data, item_cols):
    st.header(PAGE_DIM)
    items = [c for c in item_cols if c in data.columns]
    df = data[items].apply(pd.to_numeric, errors="coerce")
    kind = dm.data_type(df)
    options = {"binary": ["auto", "tetrachoric", "pearson"], "ordinal": ["auto", "polychoric", "pearson"],
               "continuous": ["auto", "pearson"]}[kind]
    st.caption(f"Algılanan veri türü: **{kind}** — yalnızca bu tür için geçerli korelasyonlar etkin.")
    c = st.columns(4)
    method = c[0].selectbox("Korelasyon", options, key="dim_method")
    rotation = c[1].selectbox("Döndürme", ["promax", "varimax"], key="dim_rot")
    nf = c[2].number_input("Faktör sayısı (0 = paralel analiz)", 0, max(len(items) - 1, 1), 0, key="dim_nf")
    reps = c[3].number_input("Paralel analiz tekrar", 10, 500, 50, key="dim_reps")
    if st.button("Boyutluluk analizini çalıştır", key="dim_run", type="primary"):
        try:
            with st.spinner("Hesaplanıyor..."):
                st.session_state["dim_result"] = dm.dimensionality_analysis(
                    df, method, int(nf) or None, rotation, int(reps))
        except ValueError as e:
            st.error(str(e)); return
    res = st.session_state.get("dim_result")
    if res is None:
        return
    cor = res["correlation"]
    m = st.columns(4)
    m[0].metric("Korelasyon", cor["method"])
    m[1].metric("KMO", f"{res['kmo']:.3f}")
    m[2].metric("Bartlett χ²", f"{res['bartlett']['χ²']:.1f}", f"sd={res['bartlett']['sd']:.0f}, p={res['bartlett']['p']:.3g}")
    m[3].metric("Paralel analiz", res["unidimensionality"]["Paralel analiz önerisi"])
    if cor["smoothed"]:
        st.warning(f"Korelasyon matrisi pozitif tanımlı değildi (en küçük özdeğer {cor['min_eigen_before']:.3g}); düzeltildi.")
    if len(cor["flags"]):
        st.warning("Polikorik/tetrakorik uyarıları"); st.dataframe(cor["flags"])
    if res["heywood"]:
        st.error("Heywood durumu: en az bir maddede h² ≥ 1.")
    if not res["paf_converged"]:
        st.warning("PAF yakınsamadı.")
    st.info(res["unidimensionality"]["Karar"])
    pa = res["parallel"]
    qcol = [c for c in pa.columns if c.startswith("Permütasyon %")][0]
    fig = go.Figure()
    fig.add_scatter(x=pa["Bileşen"], y=pa["Gözlenen özdeğer"], mode="lines+markers", name="Gözlenen")
    fig.add_scatter(x=pa["Bileşen"], y=pa[qcol], mode="lines+markers", name=qcol, line=dict(dash="dash"))
    fig.update_layout(title="Yamaç birikinti grafiği ve paralel analiz", xaxis_title="Bileşen", yaxis_title="Özdeğer")
    st.plotly_chart(fig, use_container_width=True)
    t1, t2, t3, t4 = st.tabs(["Faktör yükleri", "Tek boyutluluk", "KMO (madde)", "Korelasyon matrisi"])
    t1.dataframe(res["loadings"].style.format("{:.3f}"))
    if res["n_factors"] > 1:
        t1.caption(f"Döndürme: {res['rotation']}. Faktörler arası korelasyon:"); t1.dataframe(res["factor_correlations"].round(3))
    t2.dataframe(pd.Series(res["unidimensionality"], name="Değer").astype(str))
    t3.dataframe(res["kmo_items"].round(3))
    t4.plotly_chart(_heat(cor["R"], f"{cor['method']} korelasyon"), use_container_width=True)
    tables = {"faktor_yukleri": res["loadings"], "paralel_analiz": pa.set_index("Bileşen"),
              "kmo_madde": res["kmo_items"].to_frame(), "korelasyon": cor["R"],
              "faktor_korelasyon": res["factor_correlations"],
              "tek_boyutluluk": pd.Series(res["unidimensionality"], name="Değer").astype(str).to_frame(),
              "ozet": pd.DataFrame({"Değer": [cor["method"], res["n"], res["kmo"], *res["bartlett"].values()]},
                                   index=["Korelasyon", "N", "KMO", "Bartlett χ²", "sd", "p"]).astype(str)}
    pngs = {"scree_paralel": _png_lines(pa["Bileşen"], {"Gözlenen": pa["Gözlenen özdeğer"], qcol: pa[qcol]},
                                        "Scree / paralel analiz", "Bileşen", "Özdeğer"),
            "korelasyon_isi_haritasi": _png_heatmap(cor["R"], "Korelasyon")}
    st.subheader("Raporlar"); st.text(DIM_METHOD)
    _downloads("boyutluluk", tables, pngs, DIM_METHOD)


def render_local_fit(data, item_cols):
    st.header(PAGE_LOCAL)
    items = [c for c in item_cols if c in data.columns]
    df = data[items].apply(pd.to_numeric, errors="coerce")
    if dm.data_type(df) != "binary":
        st.error("Bu ekran yalnızca 0/1 puanlanmış veriyi destekler. Çok kategorili modeller için S-X² "
                 "uygulanmamıştır; geçersiz çıkarım üretilmemesi için analiz kapatıldı.")
        return
    c = st.columns(4)
    model = c[0].selectbox("Model", list(lf.SUPPORTED), index=1, key="lf_model")
    person = c[1].selectbox("Kişi kestirimi (Q3)", ["WLE", "EAP"], key="lf_person")
    alpha = c[2].number_input("α (FDR)", 0.001, 0.2, 0.05, key="lf_alpha")
    boot = c[3].number_input("Q3 bootstrap (0 = kapalı)", 0, 200, 0, key="lf_boot")
    if model not in lf.CERTIFIED_SX2:
        st.warning(f"{model}: S-X² p/FDR çıkarımları {lf.EXPERIMENTAL}; karar üretilmez.")
    if st.button("Analizi çalıştır", key="lf_run", type="primary"):
        try:
            with st.spinner("Kestirim ve tanılar hesaplanıyor..."):
                st.session_state["lf_result"] = lf.run(df, model, float(alpha), person, int(boot))
        except ValueError as e:
            st.error(str(e)); return
    res = st.session_state.get("lf_result")
    if res is None:
        return
    if not res["fit"]["converged"]:
        st.warning("MTK kestirimi yakınsamadı; tanılar yorumlanırken dikkat edin.")
    li = res["local"]; q3 = li["q3"]
    m = st.columns(3)
    m[0].metric("Ortalama Q3", f"{q3['mean_q3']:.3f}")
    m[1].metric("YB altında beklenen", f"{q3['expected_under_li']:.3f}")
    bs = q3.get("bootstrap")
    m[2].metric("Bootstrap %95 (maks. Q3*)", f"{bs['max_quantile_95']:.3f}" if bs else "—")
    tab1, tab2, tab3, tab4 = st.tabs(["Madde çiftleri", "Q3 ısı haritası", "S-X² madde uyumu", "Gözlenen/beklenen"])
    tab1.dataframe(li["pairs"].round(4), use_container_width=True)
    tab1.caption("Evrensel kesme puanı kullanılmaz; çiftler merkezlenmiş Q3'e göre sıralıdır.")
    tab2.plotly_chart(_heat(q3["adjusted"], "Merkezlenmiş Q3"), use_container_width=True)
    tab2.plotly_chart(_heat(li["residuals"], "Tek faktör artık korelasyonları"), use_container_width=True)
    tables = {"madde_ciftleri": li["pairs"], "q3": q3["q3"], "merkezlenmis_q3": q3["adjusted"],
              "artik_korelasyon": li["residuals"], "madde_parametreleri": res["items"]}
    pngs = {"q3_isi_haritasi": _png_heatmap(q3["adjusted"], "Merkezlenmiş Q3")}
    if res["itemfit"] is None:
        tab3.warning(res["note"])
    else:
        it = res["itemfit"]
        if not res["certified"]:
            tab3.error(f"{lf.EXPERIMENTAL}: p ve FDR değerleri karar için kullanılmamalıdır.")
        tab3.dataframe(it.round(4), use_container_width=True)
        tables["sx2_madde_uyumu"] = it
        sel = tab4.selectbox("Madde", list(res["obs_exp"]), key="lf_item")
        g = res["obs_exp"][sel]
        fig = go.Figure()
        fig.add_scatter(x=g["Grup"], y=g["Gözlenen oran"], mode="lines+markers", name="Gözlenen")
        fig.add_scatter(x=g["Grup"], y=g["Beklenen oran"], mode="lines+markers", name="Beklenen", line=dict(dash="dash"))
        fig.update_layout(title=f"{sel}: puan grubu bazında gözlenen/beklenen", xaxis_title="Toplam puan grubu (birleştirilmiş)",
                          yaxis_title="P(1)")
        tab4.plotly_chart(fig, use_container_width=True)
        tab4.dataframe(g.round(4))
        allg = pd.concat({k: v for k, v in res["obs_exp"].items()}, names=["Madde", "satır"])
        tables["gozlenen_beklenen"] = allg
        pngs[f"gozlenen_beklenen_{sel}"] = _png_lines(g["Grup"], {"Gözlenen": g["Gözlenen oran"], "Beklenen": g["Beklenen oran"]},
                                                      f"{sel} S-X² tanı", "Puan grubu", "P(1)")
    st.subheader("Raporlar"); st.text(LOCAL_METHOD)
    _downloads("yerel_bagimsizlik_madde_uyumu", tables, pngs, LOCAL_METHOD)
