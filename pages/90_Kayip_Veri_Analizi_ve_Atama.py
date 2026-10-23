"""Standalone Streamlit multipage missing-data laboratory."""
import io
import pandas as pd
import streamlit as st
from core.missing_imputation import diagnose, apply, METHODS

st.title("🧩 Kayıp Veri Analizi ve Atama")
st.caption("PsikoAnaliz • scikit-learn destekli • yalnızca seçilen madde sütunlarında işlem yapar")
st.info("MCAR/MAR/MNAR mekanizması yalnızca kayıp oranlarından kesin olarak belirlenemez. "
        "MTK için eksikleri olabilirlikte modellemek, tek atamadan daha uygun olabilir.")
source = st.file_uploader("CSV veya Excel yükleyin", type=["csv","xlsx"], key="missing_lab_file")
if source is not None:
    if source.name.lower().endswith(".csv"):
        df = pd.read_csv(source)
    else:
        df = pd.read_excel(source)
    defaults = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and
                c.lower() not in ("id", "participant_id", "score_total")]
    cols = st.multiselect("Analiz edilecek madde sütunları (ID, grup, toplam puan seçmeyin)",
                          list(df.columns), default=defaults)
    if cols:
        try:
            report = diagnose(df, cols)
            a,b,c = st.columns(3)
            a.metric("Eksik hücre", report["missing"])
            b.metric("Eksik oranı", f'{report["missing_pct"]:.2f}%')
            c.metric("Tam kayıt", report["complete_rows"])
            st.bar_chart(report["per_item"].set_index("item")["missing_rate"])
            st.dataframe(report["per_item"], use_container_width=True)
            st.warning(report["warning"])
            if report["high_missing"]:
                st.warning(">%30 kayıp madde: "+", ".join(report["high_missing"]))
            details = {
                "keep":"Kayıpları koru — atama yapmaz; MTK motorunun eksikleri doğrudan işlemesi ayrıca gerekir.",
                "listwise":"Tam kayıt — eksik satırları çıkarır; seçilim yanlılığı riski.",
                "zero":"0 puan — yalnızca atlanan yanıtın gerçekten yanlış sayıldığı puanlama kuralında.",
                "mode":"En sık kategori — ikili/sıralı kategori değerlerini korur, varyansı azaltabilir.",
                "median":"Medyan — sayısal merkezi değer; kategorik madde için dikkat.",
                "knn":"KNN — yakın yanıt örüntülerine göre atama; ikili maddelerde yuvarlama gerekir."
            }
            st.write("**Tanısal öneri:** "+details[report["recommended"]])
            method = st.radio("Uygulanacak yöntem", METHODS,
                              index=METHODS.index(report["recommended"]),
                              format_func=lambda m:details[m])
            neighbors = st.number_input("KNN komşu sayısı", 1, 30, 5) if method=="knn" else 5
            rounding = False
            if report["binary"] and method in ("median","knn"):
                rounding = st.checkbox("0.5 eşiğiyle 0/1 yuvarlamasını kabul ediyorum")
            st.caption("Tek atama standart hataları küçümseyebilir. KNN/medyan çoklu atama değildir.")
            if st.button("▶ Seçili yöntemi uygula", type="primary"):
                result = apply(df,cols,method,neighbors=int(neighbors),binary_round=rounding)
                st.session_state["missing_lab_result"] = result
                st.session_state["missing_lab_method"] = method
            if "missing_lab_result" in st.session_state:
                result = st.session_state["missing_lab_result"]
                st.success(f"İşlem tamamlandı: {len(result)} satır; yöntem {st.session_state['missing_lab_method']}.")
                st.dataframe(result.head(20),use_container_width=True)
                st.download_button("⬇ İşlenmiş CSV indir",
                                   result.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="psikoanaliz_kayip_veri_islenmis.csv",
                                   mime="text/csv")
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
