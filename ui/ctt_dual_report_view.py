import streamlit as st
from core.ctt_dual_engine import compare_ctt,item_guidance
from reports.ctt_guided_docx import ctt_docx

def render(data,items=(),id_col=None,group_col=None):
    st.subheader("KTK: Çift Motor ve Akademik Rapor")
    choices=[c for c in data.columns if c not in (id_col,group_col)]
    cols=st.multiselect("Madde sütunları",choices,default=[c for c in items if c in choices] or choices[:10],key="ctt_double_columns")
    if st.button("KTK raporu hesapla",key="ctt_double_run"):
        try:st.session_state["ctt_double_result"]=compare_ctt(data,cols)
        except Exception as exc:st.error(str(exc))
    r=st.session_state.get("ctt_double_result")
    if r is None:return
    st.write("Yerel alfa:",r["native_alpha"],"Pingouin alfa:",r["reference_alpha"])
    st.dataframe(item_guidance(r),hide_index=True)
    st.markdown("**Bağımsız madde istatistikleri karşılaştırması**")
    st.dataframe(r["item_crosscheck"],hide_index=True)
    st.download_button("Karşılaştırma CSV",r["item_crosscheck"].to_csv(index=False).encode("utf-8-sig"),"KTK_karsilastirma.csv",mime="text/csv",key="ctt_cross_csv")
    st.info("Madde kararları uzman incelemesi gerektirir; otomatik silme yapılmaz.")
    st.download_button("KTK Word raporu",ctt_docx(r),"KTK_Akademik_Rapor.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",key="ctt_double_docx")
