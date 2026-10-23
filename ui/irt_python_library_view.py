"""Direct Python mirt analysis entry, distinct from EserPsychometrics native pages."""
import pandas as pd
import streamlit as st
from core.irt_python_mirt_adapter import fit_python_mirt

MODELS=("1PL","2PL","3PL","4PL","GRM","PCM","GPCM","RSM","NRM")

def render(data,items=(),id_col=None,group_col=None,model="1PL"):
    st.subheader(f"Python mirt Kütüphanesi — {model}")
    st.caption("Bağımsız Python mirt 1.2.0 motoru. Bu ekran yerel motorla aynı "
               "parametreleri veya tam bilimsel eşdeğerliği garanti etmez; "
               "kestirim ve yakınsama bilgileri incelenmelidir.")
    choices=[c for c in data.columns if c not in (id_col,group_col)]
    cols=st.multiselect("Madde sütunları",choices,default=[c for c in items if c in choices] or choices[:10],
                         key=f"python_mirt_cols_{model}")
    categories=None
    if model in ("GRM","PCM","GPCM","RSM","NRM"):
        categories=st.number_input("Kategori sayısı",min_value=3,max_value=20,value=3,step=1,
                                   key=f"python_mirt_categories_{model}")
    if st.button("Python mirt ile hesapla",key=f"python_mirt_start_{model}"):
        try:
            if len(cols)<3:raise ValueError("En az üç madde seçin.")
            x=data[cols].apply(pd.to_numeric,errors="raise")
            if x.isna().any().any():raise ValueError("Eksik yanıtları bu modülde önce temizleyin.")
            with st.spinner("Python mirt kestirimi çalışıyor"):
                outcome=fit_python_mirt(x.to_numpy(),model=model,n_categories=int(categories) if categories else None)
            st.session_state[f"python_mirt_fit_{model}"]=(outcome,tuple(cols),pd.util.hash_pandas_object(x,index=True).sum())
        except Exception as exc:st.error(f"Kestirim yapılamadı: {exc}")
    result=st.session_state.get(f"python_mirt_fit_{model}")
    if result is None:return
    outcome,old_cols,old_hash=result
    try:current_hash=pd.util.hash_pandas_object(data[list(old_cols)],index=True).sum()
    except Exception:return
    if tuple(cols)!=old_cols or current_hash!=old_hash:
        st.info("Veri veya madde seçimi değişti. Yeniden hesaplayın.");return
    st.success(f"Python mirt kestirimi çalıştırıldı: N={outcome.n_persons}, k={outcome.n_items}.")
    st.warning("Bilimsel statü: bağımsız model bazlı doğrulama gerekli. "
               "Bu sonuç formel DIF değildir.")
    fit=outcome.fit
    if hasattr(fit,"parameters"):
        p=getattr(fit,"parameters")
        st.write("Kestirilen parametreler:")
        st.dataframe(pd.DataFrame(p) if not isinstance(p,pd.DataFrame) else p)
    else:
        st.info("Bu model için kararlı genel parametre tablosu API'si doğrulanmadı. "
                "Ham nesne raporu yayınlanmıyor.")
