"""Direct Python mirt analysis entry, distinct from EserPsychometrics native pages."""
import hashlib
import pandas as pd
import streamlit as st
from core.irt_python_mirt_adapter import fit_python_mirt
from ui.irt_solver_controls import solver_controls

MODELS=("Rasch","1PL","2PL","3PL","4PL","GRM","PCM","GPCM","NRM")

def _data_fingerprint(frame):
    """Order-sensitive digest of data, columns, index and dtype."""
    hashes = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    info = repr((tuple(frame.columns), tuple(map(str,frame.dtypes)),frame.shape)).encode("utf-8")
    return hashlib.sha256(info + hashes).hexdigest()


def render(data,items=(),id_col=None,group_col=None,model="1PL"):
    st.subheader(f"Python mirt Kütüphanesi — {model}")
    st.caption("Bağımsız Python mirt 1.2.0 motoru. Bu ekran yerel motorla aynı "
               "parametreleri veya tam bilimsel eşdeğerliği garanti etmez; "
               "kestirim ve yakınsama bilgileri incelenmelidir.")
    choices=[c for c in data.columns if c not in (id_col,group_col)]
    cols=st.multiselect("Madde sütunları",choices,default=[c for c in items if c in choices] or choices[:10],
                         key=f"python_mirt_cols_{model}")
    settings=solver_controls(f"python_mirt_solver_{model}")
    if model=="Rasch":st.caption("Python Rasch: a=1, latent N(0,1). Yerel serbest-SD Rasch tanımıyla karıştırmayın.")
    categories=None
    if model in ("GRM","PCM","GPCM","RSM","NRM"):
        categories=st.number_input("Kategori sayısı",min_value=3,max_value=20,value=3,step=1,
                                   key=f"python_mirt_categories_{model}")
    if st.button("Python mirt ile hesapla",key=f"python_mirt_start_{model}"):
        st.session_state.pop(f"python_mirt_fit_{model}", None)
        try:
            if len(cols)<3:raise ValueError("En az üç madde seçin.")
            x=data[cols].apply(pd.to_numeric,errors="raise")
            if x.isna().any().any():raise ValueError("Eksik yanıtları bu modülde önce temizleyin.")
            with st.spinner("Python mirt kestirimi çalışıyor"):
                outcome=fit_python_mirt(x.to_numpy(),model=model,n_categories=int(categories) if categories else None,**settings)
            st.session_state[f"python_mirt_fit_{model}"]=(outcome,tuple(cols),_data_fingerprint(data),int(categories) if categories else None,settings)
        except Exception as exc:st.error(f"Kestirim yapılamadı: {exc}")
    result=st.session_state.get(f"python_mirt_fit_{model}")
    if result is None:return
    if len(result)!=5:
        st.session_state.pop(f"python_mirt_fit_{model}",None);return
    outcome,old_cols,old_hash,old_categories,old_settings=result
    try:current_hash=_data_fingerprint(data)
    except Exception:
        st.session_state.pop(f"python_mirt_fit_{model}", None)
        st.warning("Veri artık okunamıyor; önceki sonuç temizlendi.")
        return
    if settings!=old_settings or tuple(cols)!=old_cols or current_hash!=old_hash or (int(categories) if categories else None)!=old_categories:
        st.session_state.pop(f"python_mirt_fit_{model}", None)
        st.info("Veri, madde veya kategori ayarları değişti. Yeniden hesaplayın.")
        return
    st.success(f"Python mirt kestirimi çalıştırıldı: N={outcome.n_persons}, k={outcome.n_items}.")
    st.warning("Bilimsel statü: bağımsız model bazlı doğrulama gerekli. "
               "Bu sonuç formel DIF değildir.")
    fit=outcome.fit
    if hasattr(fit,"parameters") and getattr(fit,"parameters") is not None:
        p=getattr(fit,"parameters")
        st.write("Kestirilen parametreler:")
        st.dataframe(pd.DataFrame(p) if not isinstance(p,pd.DataFrame) else p)
    else:
        st.info("Bu model için kararlı genel parametre tablosu API'si doğrulanmadı. "
                "Ham nesne raporu yayınlanmıyor.")

    st.caption(f"Uygulanan çözücü ayarları: {outcome.solver_settings}")
    if model=="Rasch":
        try:
            from core.irt_rasch_wright import rasch_wright_tables,wright_figure,wright_exports
            tables=rasch_wright_tables(outcome,data[cols],item_names=cols,
                identifiers=data[id_col].to_numpy() if id_col in data.columns else None)
            st.caption('EAP kişi yetenekleri ve b madde konumları aynı N(0,1) logit ölçeğindedir. EAP posterior SD, bilgi SEM’i değildir.')
            st.plotly_chart(wright_figure(tables),use_container_width=True)
            csv,excel=wright_exports(tables)
            for name,table in tables.items():
                st.dataframe(table,hide_index=True)
                st.download_button(f'Wright {name} — CSV',csv[name],f'Rasch_wright_{name}.csv','text/csv',key=f'rasch_wright_{name}',on_click='ignore')
            st.download_button('Wright tabloları — Excel',excel,'Rasch_wright.xlsx',key='rasch_wright_excel',on_click='ignore')
        except Exception as exc:st.warning(f'Wright haritası sunulmadı: {exc}')
