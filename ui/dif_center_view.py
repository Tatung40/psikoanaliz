"""DIF center: shared observed-score report plus existing formal routes."""
import io
import pandas as pd
import streamlit as st
from core.dif_center import METHODS,LIMITATIONS,observed,support_matrix,theta_sensitivity,fingerprint
from core.reproducible import displayed
from reports.dif_center_report import downloads,figure
from reports.irt_b14_report import safe_frame

KEY='dif_center_result'


def render(data=None,item_cols=(),id_col=None,group_col=None):
    st.subheader('⚖️ DIF Analiz Merkezi')
    st.warning(LIMITATIONS)
    with st.expander('Yedi yöntem — gerçek destek ve kanıt durumu',expanded=True):
        st.dataframe(support_matrix(),hide_index=True)
        st.caption('Uysal ve ark. (2019), DOI 10.21031/epod.534312: PDF bu oturumda sağlanmadı; makaleyle eşdeğerlik iddiası yok.')
    uploaded=st.file_uploader('DIF merkezi verisi (CSV/XLSX)',type=['csv','xlsx'],key='dc_file')
    if uploaded:
        try:data=pd.read_excel(io.BytesIO(uploaded.getvalue())) if uploaded.name.lower().endswith('.xlsx') else pd.read_csv(io.BytesIO(uploaded.getvalue()))
        except Exception as exc:
            st.session_state.pop(KEY,None);st.error(f'Veri okunamadı: {exc}');return
    if data is None or data.empty:
        st.session_state.pop(KEY,None);st.info('Veri yükleyin.');return
    route=st.radio('Analiz yolu',['MH / Lojistik / SIBTEST','IRT-LR / Raju alanı',
        'Çapa arındırması / ordinal gözlenen puan','GRM formel DIF','PCM / GPCM / RSM / NRM formel DIF'],key='dc_route')
    if route=='IRT-LR / Raju alanı':
        from ui.python_mirt_dif_view import render as formal
        formal(data,item_cols,id_col,group_col);return
    if route=='Çapa arındırması / ordinal gözlenen puan':
        from ui.dif_classical_view import render as classical
        classical(data,item_cols,id_col,group_col);return
    if route=='GRM formel DIF':
        from ui.grm_dif_view import render as grm
        grm(data,item_cols,id_col,group_col);return
    if route=='PCM / GPCM / RSM / NRM formel DIF':
        from ui.polytomous_dif_view import render as poly
        poly(data,item_cols,id_col,group_col);return
    group=st.selectbox('Grup sütunu',list(data.columns),index=list(data.columns).index(group_col) if group_col in data else 0,key='dc_group')
    levels=list(pd.Series(data[group]).dropna().unique())
    if len(levels)<2:st.session_state.pop(KEY,None);st.error('İki grup düzeyi gerekli.');return
    reference=st.selectbox('Referans',levels,key='dc_ref')
    focal=st.selectbox('Odak',[v for v in levels if v!=reference],key='dc_focal')
    choices=[c for c in data.columns if c not in (group,id_col)]
    items=st.multiselect('İkili maddeler',choices,default=[c for c in item_cols if c in choices] or choices,key='dc_items')
    anchors=st.multiselect('Önceden belirlenmiş çapalar',items,default=items[:-2],key='dc_anchors')
    st.caption('Varsayılan çapa seçimi değişmezlik kanıtı değildir; bilimsel gerekçeyle seçin. Çapalar test ailesinden ayrıdır.')
    studied=st.multiselect('Ortak test ailesi',[c for c in items if c not in anchors],
        default=[c for c in items if c not in anchors],key='dc_studied')
    methods=st.multiselect('Yöntemler',METHODS,default=list(METHODS),key='dc_methods')
    correction=st.selectbox('Yöntem içi düzeltme',('Holm','Bonferroni','BH','none'),key='dc_correction')
    alpha=st.number_input('α',min_value=.001,max_value=.2,value=.05,key='dc_alpha')
    lr_test=st.selectbox('Birincil lojistik hipotezi',('omnibus','uniform','nonuniform'),key='dc_lr_test')
    sib_method=st.selectbox('SIBTEST hipotezi',('original','crossing'),key='dc_sib_method')
    sib_correction=st.checkbox('SIBTEST gerçek-puan regresyon düzeltmesi',value=True,key='dc_sib_correction')
    settings=(items,group,reference,focal,anchors,studied,methods,correction,alpha,lr_test,sib_method,sib_correction,id_col)
    key=fingerprint(data,settings)
    saved=st.session_state.get(KEY)
    if saved and saved['key']!=key:st.session_state.pop(KEY,None);saved=None
    if st.button('DIF merkezini çalıştır',type='primary',key='dc_run'):
        st.session_state.pop(KEY,None);saved=None
        try:
            result=observed(data,items=items,group=group,reference=reference,focal=focal,anchors=anchors,
                studied=studied,methods=methods,correction=correction,alpha=alpha,lr_test=lr_test,
                sib_method=sib_method,sib_correction=sib_correction)
            saved={'key':key,'result':result};st.session_state[KEY]=saved
        except Exception as exc:st.error(f'Analiz başarısız: {exc}')
    if saved is None:return
    result=saved['result'];displayed(result)
    st.dataframe(safe_frame(result['table']),hide_index=True)
    st.caption(result['metadata']['Matching'])
    st.json(result['metadata'])
    fig=figure(result);st.pyplot(fig)
    import matplotlib.pyplot as plt
    plt.close(fig)
    with st.expander('Madde çıkarma / koruma — sabit kalibrasyonda θ duyarlılığı'):
        removed=st.multiselect('Puanlamadan çıkarılacak maddeler',items,key='dc_removed')
        model=st.selectbox('Puanlama modeli',('Rasch','1PL','2PL'),key='dc_score_model')
        theta_key=fingerprint(data,(key,removed,model))
        if result.get('theta_key')!=theta_key:
            result.pop('theta',None);result.pop('theta_key',None)
            result['metadata'].pop('Theta_removed',None);result['metadata'].pop('Theta_model',None)
        st.caption('Tek havuzlanmış ML kalibrasyonu, GH61 EAP. Aynı parametreler/ölçek korunur; yeniden kalibrasyon ve gerçek θ/RMSD yok. Seçim sonrası betimsel duyarlılık; yanlılık düzeltmesi değildir.')
        if st.button('θ duyarlılığını hesapla',key='dc_theta_run'):
            try:
                selected=data.loc[data[group].isin([reference,focal])]
                result['theta']=theta_sensitivity(selected,items=items,removed=removed,model=model,
                    identifiers=selected[id_col].to_numpy() if id_col in selected else selected.index.to_numpy())
                result['theta_key']=theta_key
                result['metadata']['Theta_removed']=removed;result['metadata']['Theta_model']=model
            except Exception as exc:st.error(f'θ duyarlılığı hesaplanamadı: {exc}')
        if 'theta' in result:
            st.dataframe(safe_frame(result['theta']),hide_index=True)
            st.download_button('θ duyarlılığı — CSV',safe_frame(result['theta']).to_csv(index=False).encode('utf-8-sig'),
                               'dif_theta_sensitivity.csv','text/csv',key='dc_theta_csv',on_click='ignore')
    payload=downloads(result)
    for name,mime,suffix in [('csv','text/csv','csv'),('xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet','xlsx'),
                             ('zip','application/zip','zip'),('png','image/png','png'),('report','text/markdown','md')]:
        st.download_button(f'DIF — {name.upper()}',payload[name],f'dif_center.{suffix}',mime,key='dc_'+name,on_click='ignore')
