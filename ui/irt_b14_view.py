"""B14 polytomous pilot: ordinal GRM/PCM/GPCM/RSM or nominal NRM.

No scoring key is inferred from raw nominal responses. Results from
Python mirt are reported as experimental, never mixed with native B13.
"""
import io
import pandas as pd
import streamlit as st
from core.irt_b14 import MODELS,ORDINAL,prepare_polytomous,fit_b14

def render(data,existing_item_cols=(),id_col=None,group_col=None):
    st.subheader("B14 — Çok Kategorili MTK (Deneysel Pilot)")
    st.caption("Sıralı kategoriler: GRM, PCM, GPCM, RSM | Sırasız A–E seçenekleri: NRM. "
               "Python mirt ayrı bir kestirim motorudur; henüz bağımsız sayısal doğrulaması tamamlanmadı.")
    model=st.radio("Model",MODELS,horizontal=True,key="b14_model")
    available=[c for c in data.columns if c not in (id_col,group_col)]
    default=[c for c in existing_item_cols if c in available]
    cols=st.multiselect("Madde sütunları",available,default=default or available[:min(10,len(available))],key="b14_items")
    if model=="NRM":
        labels=st.text_input("Nominal seçenekler (virgülle)",value="A,B,C,D,E",key="b14_categories")
        categories=tuple(v.strip().upper() for v in labels.split(","))
        st.info("NRM doğru cevabı zorunlu kılmaz: A–E seçeneklerinin her biri ayrı kategoridir. "
                "Doğru anahtarına göre 0/1 yapmak istiyorsanız B13 ikili MTK kullanın.")
    else:
        categories=None
        st.info("GRM/PCM/GPCM/RSM için 0,1,2,3… gibi sıralı kısmi puanlar gerekli. "
                "A–E seçenek etiketleri bu modellerde başarı düzeyi olarak varsayılmaz.")
    result=None
    if not cols or len(cols)<3:
        st.warning("En az üç madde seçin.")
    else:
        subset=data.loc[:,cols]
        try:
            numeric,category_labels=prepare_polytomous(subset,model,categories=categories)
            st.success(f"Veri uygunluğu: {len(numeric)} kişi, {numeric.shape[1]} madde, "
                       f"{len(category_labels)} kategori.")
            counts=numeric.stack().value_counts().sort_index().rename_axis("Kategori kodu").reset_index(name="Frekans")
            counts["Etiket"]=counts["Kategori kodu"].map(dict(enumerate(category_labels)))
            st.dataframe(counts,hide_index=True,use_container_width=True)
        except ValueError as e:
            numeric=None
            st.error(f"Veri denetimi: {e}")
        if st.button("Python mirt ile B14 pilot analizi",key="b14_run",type="primary",disabled=numeric is None):
            try:
                with st.spinner("Çok kategorili model kalibre ediliyor"):
                    result=fit_b14(subset,model,categories=categories)
                st.session_state["b14_current"]={"model":model,"columns":tuple(cols),
                    "datahash":int(pd.util.hash_pandas_object(subset,index=True).sum()),
                    "result":result}
            except Exception as e:
                st.error(f"Kestirim başarısız: {e}")
        saved=st.session_state.get("b14_current")
        if saved and saved["model"]==model and saved["columns"]==tuple(cols) and saved["datahash"]==int(pd.util.hash_pandas_object(subset,index=True).sum()):
            result=saved["result"]
    if result is not None:
        st.warning(result["status"])
        fit=result["result"]
        st.write("Model:",model,"| Motor:",result["engine"],"| N:",result["N"],"| Madde:",result["k"])
        try:
            summary=fit.coef_with_se()
            st.dataframe(summary,hide_index=False,use_container_width=True)
            st.download_button("Madde parametrelerini CSV indir",summary.to_csv(index=True).encode("utf-8-sig"),
                               "B14_pilot_parametreler.csv",mime="text/csv",key="b14_csv")
        except Exception as exc:
            st.info(f"Standartlaştırılmış parametre tablosu alınamadı: {exc}")
        st.caption("Parametre SH, model uyumu ve karşılaştırma yorumları bağımsız doğrulama tamamlanmadan "
                   "kesin bilimsel kanıt olarak sunulmaz.")
