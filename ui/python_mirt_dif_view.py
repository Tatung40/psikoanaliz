"""Explicit public-library multigroup DIF route, isolated from local DIF engines."""
import io
import pandas as pd
import streamlit as st
from core.reproducible import displayed
from core.python_mirt_dif import MODELS,FAMILIES,LIMITATION,analyze,fingerprint,prepare,support_matrix
from reports.python_mirt_dif_report import downloads,figure
from reports.dif_academic_docx import dif_academic_docx
from ui.irt_solver_controls import solver_controls

KEY='python_mirt_dif_result'

def render(data=None,item_cols=(),id_col=None,group_col=None):
    st.subheader('⚖️ Python mirt — DIF Analizleri')
    st.warning(LIMITATION)
    with st.expander('Doğrulanmış API destek matrisi'):
        st.dataframe(support_matrix(),hide_index=True)
        st.caption('RSM, Wald, c/u DIF, otomatik çapa seçimi ve ortak-serbest-a 1PL desteklenmiyor. RSM seçeneği kapalıdır.')
    uploaded=st.file_uploader('DIF verisi (CSV/XLSX); mevcut yükleme de kullanılabilir',type=['csv','xlsx'],key='pmd_file')
    if uploaded:
        try:
            data=pd.read_excel(io.BytesIO(uploaded.getvalue())) if uploaded.name.lower().endswith('.xlsx') else pd.read_csv(io.BytesIO(uploaded.getvalue()))
        except Exception as exc:
            st.session_state.pop(KEY,None);st.error(f'Veri okunamadı: {exc}');return
    if data is None or data.empty:
        st.session_state.pop(KEY,None);st.info('Önce veri yükleyin.');return
    candidates=[c for c in data.columns if c!=id_col]
    if not candidates:st.session_state.pop(KEY,None);st.error('Grup ve madde sütunları gerekli');return
    group=st.selectbox('Grup sütunu',candidates,index=candidates.index(group_col) if group_col in candidates else 0,key='pmd_group')
    levels=data[group].dropna().unique().tolist()
    if len(levels)<2:st.session_state.pop(KEY,None);st.error('En az iki grup düzeyi gerekli');return
    reference=st.selectbox('Referans grup',levels,key='pmd_reference')
    focal=st.selectbox('Odak grup',[v for v in levels if v!=reference],key='pmd_focal')
    choices=[c for c in candidates if c!=group]
    items=st.multiselect('Maddeler',choices,default=[c for c in item_cols if c in choices] or choices,key='pmd_items')
    model=st.selectbox('Model',MODELS,key='pmd_model')
    if model in ('1PL','Rasch'):st.info('Bu public çok gruplu API’de a=1; ortak serbest a 1PL değildir.')
    labels=None
    if model=='NRM':
        observed=list(dict.fromkeys(str(v).strip().upper() for v in data[items].to_numpy().ravel())) if items else []
        labels=tuple(v.strip().upper() for v in st.text_input('Nominal etiketler (açık kod sırası; başarı sırası değildir)',value=','.join(observed),key='pmd_labels').split(','))
    families=({'Kesişim ailesi (nominal)':('intercepts',),'Eğim ailesi (nominal)':('discrimination',),'Omnibus (nominal)':('discrimination','intercepts')} if model=='NRM' else FAMILIES)
    options=['Uniform / konum'] if model in ('Rasch','1PL','PCM') else list(families)
    test=st.selectbox('LR parametre ailesi',options,key='pmd_test')
    anchors=st.multiselect('Önceden belirlenmiş çapa maddeleri',items,default=items[:2],key='pmd_anchors')
    studied_options=[c for c in items if c not in anchors]
    studied=st.multiselect('İncelenen maddeler',studied_options,default=studied_options,key='pmd_studied')
    correction=st.selectbox('Talep edilen düzeltme (çıkarım açılana kadar uygulanmaz)',('fdr_bh','holm','bonferroni','none'),key='pmd_correction')
    alpha=st.number_input('Alpha (çıkarım kapalı)',min_value=.001,max_value=.2,value=.05,step=.01,key='pmd_alpha')
    sensitivity=st.checkbox('İkinci açık çapa tasarımıyla duyarlılık',key='pmd_sensitivity')
    alternative=st.multiselect('İkinci çapa seti',[c for c in items if c not in studied],default=anchors,key='pmd_alternative') if sensitivity else []
    solver=solver_controls('pmd_solver')
    settings=(tuple(items),group,reference,focal,model,tuple(labels or ()),test,tuple(anchors),tuple(studied),
              correction,alpha,sensitivity,tuple(alternative),tuple(sorted(solver.items())))
    key=fingerprint(data,settings)
    saved=st.session_state.get(KEY)
    if saved is not None and (not isinstance(saved,dict) or saved.get('key')!=key):
        st.session_state.pop(KEY,None);saved=None
        st.info('Veri veya ayarlar değişti; eski sonuçlar ve indirmeler temizlendi.')
    try:
        _,_,_,freq,excluded=prepare(data,items,group,reference,focal,model,labels)
        if len(anchors)<2 or not studied:raise ValueError('En az iki açık çapa ve bir incelenen madde gerekli')
        if sensitivity and (len(alternative)<2 or set(alternative)==set(anchors)):
            raise ValueError('Duyarlılık için en az iki çapalı farklı bir set seçin')
    except Exception as exc:
        st.session_state.pop(KEY,None);st.error(f'Ön kontrol: {exc}');return
    with st.expander('Grup/kategori ön kontrolü'):
        st.dataframe(freq,hide_index=True);st.caption(f'Seçilmeyen grup satırları: {excluded}. Eksik yanıtlar otomatik silinmez.')
        if (freq.Frekans<5).any():st.warning('Seyrek kategoriler (<5); bu bir yeterlilik eşiği veya p değeri onayı değildir.')
    st.caption(f'{len(studied)+1} kütüphane kestirimi'+(' + ikinci çapa tasarımı' if sensitivity else '')+'; add tasarımı, manuel çapalar, otomatik arındırma yok.')
    if st.button('Python mirt DIF çalıştır',type='primary',key='pmd_run'):
        st.session_state.pop(KEY,None);saved=None
        try:
            args=dict(items=items,group=group,reference=reference,focal=focal,model=model,studied=studied,
                      parameters=families[test],correction=correction,alpha=alpha,labels=labels,**solver)
            with st.spinner('Gerçek Python mirt çok gruplu ML kestirimi'):
                result=analyze(data,anchors=anchors,**args)
                if sensitivity:
                    second=analyze(data,anchors=alternative,**args)
                    for name,frame in second['tables'].items():result['tables']['Alt_'+name]=frame
                    result['metadata']['Alternative_anchors']=alternative
                    a=result['tables']['DIF'][['Item','LR']].rename(columns={'LR':'Primary_LR'})
                    b=second['tables']['DIF'][['Item','LR']].rename(columns={'LR':'Alternative_LR'})
                    comparison=a.merge(b,on='Item',validate='one_to_one')
                    comparison['LR_change']=comparison.Alternative_LR-comparison.Primary_LR
                    result['tables']['Sensitivity']=comparison
            saved=dict(key=key,result=result);st.session_state[KEY]=saved
        except Exception as exc:st.error(f'Kestirim başarısız: {exc}')
    if not saved:return
    result=saved['result'];tables=result['tables'];displayed(result)
    if not tables['Fit'].Converged.all():st.error('NOT CONVERGED — çıktılar geçici, çıkarım kapalı.')
    st.info('NOT CERTIFIED — p/q ve kararlar kapalı; betimsel LR ve kütüphane eğrileri.')
    for title,name in [('Analiz özeti','Summary'),('Model uyumu / yakınsama','Fit'),('Grup dağılımları','Latent'),('Madde bazında DIF','DIF'),('Parametre farkları','Contrasts'),('Betimsel olasılık etkisi','Effect')]:
        st.markdown('#### '+title);st.dataframe(tables[name],hide_index=True,use_container_width=True)
    with st.expander('Tam tablolar ve çapa duyarlılığı'):
        for name,frame in tables.items():
            if name.startswith('Alt_') or name in ('Parameters','Categories','DTF','Sensitivity'):
                st.write(name);st.dataframe(frame,hide_index=True)
    item=st.selectbox('Grafik maddesi',items,key='pmd_plot_item')
    fig=figure(tables,item);st.pyplot(fig)
    import matplotlib.pyplot as plt
    plt.close(fig)
    bundle_key=(key,item)
    if saved.get('bundle_key')!=bundle_key:
        saved['downloads']=downloads(result,item);saved['bundle_key']=bundle_key
    bundle=saved['downloads']
    try:
        docx_bytes=dif_academic_docx(result,item)
        st.download_button('Akademik DIF raporu — Word',docx_bytes,
            'python_mirt_dif_academic.docx',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            key='pmd_download_academic_docx')
    except (ValueError,KeyError,TypeError) as exc:
        st.warning(f'Word raporu üretilemedi: {exc}')
    for label,ext,mime in [('DIF — CSV','csv','text/csv'),('Tam rapor — XLSX','xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),('Grafikler — PNG','png','image/png'),('Tüm sonuçlar — ZIP','zip','application/zip'),('Yöntem ve sınırlılıklar — Markdown','report','text/markdown')]:
        st.download_button(label,bundle[ext],'python_mirt_dif.'+('md' if ext=='report' else ext),mime,key='pmd_download_'+ext)
