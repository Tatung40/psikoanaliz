"""Shared MTK assumptions page for dichotomous and polytomous data."""
import pandas as pd
import streamlit as st
from core.irt_assumptions import preflight

def render(data,item_cols=(),id_col=None,group_col=None):
    st.subheader("🔍 MTK Varsayım Denetimi")
    st.caption("İkili ve çok kategorili MTK analizleri öncesinde bilimsel ön kontrol.")
    cols=[c for c in data.columns if c not in (id_col,group_col)]
    default=[c for c in item_cols if c in cols] or cols
    chosen=st.multiselect("İncelenecek madde sütunları",cols,default=default,key="irt_check_items")
    mode=st.radio("Yanıt türü",["İkili (0/1)","Sıralı (0,1,2,...)","Nominal (A,B,C,D,E)"],
                  key="irt_check_kind")
    if not chosen:
        st.warning("Madde sütunlarını seçin.")
        return
    raw=data[chosen]
    if mode.startswith("Nominal"):
        labels=st.text_input("Yanıt seçenekleri",value="A,B,C,D,E",key="irt_check_labels")
        mapping={v.strip().upper():i for i,v in enumerate(labels.split(",")) if v.strip()}
        x=raw.astype(str).apply(lambda s:s.str.strip().str.upper().map(mapping))
        st.caption("Nominal seçeneklerin kodları yalnızca hesaplamaya yöneliktir; A–E doğru/yanlış ya da başarı sırası değildir.")
        kind="nominal"
    else:
        x=raw.apply(pd.to_numeric,errors="coerce")
        kind="binary" if mode.startswith("İkili") else "ordinal"
    if st.button("🧪 Varsayımları kontrol et",key="irt_check_run",type="primary"):
        report=preflight(x,kind=kind)
        st.session_state["irt_assumption_last"]={
            "mode":mode,"columns":tuple(chosen),
            "data_hash":int(pd.util.hash_pandas_object(raw,index=True).sum()),
            "report":report}
    saved=st.session_state.get("irt_assumption_last")
    if not saved or saved["mode"]!=mode or saved["columns"]!=tuple(chosen) or saved["data_hash"]!=int(pd.util.hash_pandas_object(raw,index=True).sum()):
        st.info("Yukarıdaki düğmeyle ön denetimi başlatın.")
        return
    report=saved["report"]
    st.markdown("### Sonuç: "+report["decision"])
    if report["blocked"]:st.error(report["explanation"])
    else:st.info(report["explanation"])
    st.dataframe(report["table"],hide_index=True,use_container_width=True)
    st.download_button("Denetim tablosunu CSV indir",
        report["table"].to_csv(index=False).encode("utf-8-sig"),
        "MTK_Varsayim_Denetimi.csv",mime="text/csv",key="irt_check_csv")
    st.markdown("**Sonraki karar:** "+(
        "Veri sorunlarını giderip denetimi yeniden çalıştırın." if report["blocked"] else
        "Kestirim yapılabilir; fakat madde ve model uyumu ile yerel bağımsızlığı model sonrası ayrıca kontrol edin."))
