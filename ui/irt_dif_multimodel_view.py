"""Universal MTK model-family differential functioning *screening* interface."""
import pandas as pd
import streamlit as st
from core.reproducible import displayed
from core.irt_dif_multimodel_screen import MODELS,screen_mtk_groups

def render(data,item_cols=(),id_col=None,group_col=None):
    st.subheader("MTK: Çok Modelli Grup Farkı ve DIF Ön Taraması")
    st.warning("Bu sayfa 2PL/3PL/4PL/GRM/PCM/GPCM/RSM/NRM için biçimsel MTK-DIF kestirimi "
        "DEĞİLDİR. Rasch için ayrı doğrulanmış ortak çok gruplu DIF analizi mevcuttur. "
        "Buradaki testler koşullu gözlenen puan farkları veya nominal ham dağılım taramasıdır.")
    model=st.selectbox("Model ailesi",MODELS,key="dif_multi_model")
    cols=[c for c in data.columns if c not in (id_col,group_col)]
    chosen=st.multiselect("Maddeler",cols,default=[c for c in item_cols if c in cols] or cols[:10],key="dif_multi_items")
    group_options=[c for c in data.columns if c not in chosen and c!=id_col]
    if not group_options:
        st.info("Madde ve kimlik dışı bir grup sütunu seçin.");return
    group=st.selectbox("Grup değişkeni",group_options,
                       index=group_options.index(group_col) if group_col in group_options else 0,
                       key="dif_multi_group")
    levels=[v for v in data[group].dropna().unique().tolist()]
    if len(levels)<2:st.warning("En az iki grup gerekli.");return
    ref=st.selectbox("Referans",levels,key="dif_multi_ref")
    focus=st.selectbox("Odak",[v for v in levels if v!=ref],key="dif_multi_focus")
    if st.button("Grup farklılık taramasını çalıştır",key="dif_multi_start"):
        try:
            with st.spinner("Gözlenen puan taraması hesaplanıyor"):
                out=screen_mtk_groups(data,chosen,data[group],ref,focus,model)
            st.session_state["dif_multi_result"]=out
        except Exception as exc:st.error(str(exc))
    result=st.session_state.get("dif_multi_result")
    if result is not None:
        displayed(result)
        st.dataframe(result,hide_index=True)
        st.download_button("Tarama tablosunu CSV indir",result.to_csv(index=False).encode("utf-8-sig"),
            "MTK_DIF_OnTarama.csv",mime="text/csv",key="dif_multi_download")
        st.caption("Ham fark/DIF taraması madde yanlılığı kanıtı değildir. Çoklu test düzeltmesi, "
                   "ölçek eşitleme, çapa kararlılığı ve model-temelli MTK-DIF ayrı doğrulama gerektirir.")
