"""Nine explicit Python mirt model routes sharing verified result reporting."""
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from core.irt_python_mirt_adapter import fit_python_mirt
from core.irt_python_mirt_report import MODELS,BINARY,build_report
from core.irt_b14 import prepare_polytomous
from reports.irt_b14_report import safe_frame


def _data_fingerprint(frame):
    """Order-sensitive digest of actual data, columns, index and dtype."""
    hashes=pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes()
    info=repr((tuple(frame.columns),tuple(map(str,frame.dtypes)),frame.shape)).encode('utf-8')
    return hashlib.sha256(info+hashes).hexdigest()


def _csv(frame):return safe_frame(frame).to_csv(index=False).encode('utf-8-sig')


def render(data,items=(),id_col=None,group_col=None,model='1PL'):
    if model not in MODELS:st.error('Desteklenmeyen model');return
    st.subheader(f'Python mirt Kütüphanesi — {model}')
    st.caption('Python mirt 1.2.0 | Hesaplanmış betimsel çıktılar; bağımsız bilimsel sertifika veya formel DIF değildir.')
    choices=[c for c in data.columns if c not in (id_col,group_col)]
    cols=st.multiselect('Madde sütunları',choices,default=[c for c in items if c in choices] or choices[:10],key=f'python_mirt_cols_{model}')
    estimation='ML';categories=None;labels=()
    if model=='Rasch':st.info('Rasch: a=1; θ~N(0,1). Yerel serbest-σ Rasch ekranından ayrıdır.')
    if model=='1PL':st.info('1PL: ortak a serbest; 2PL üzerinde belgelenmiş eşitlik kısıtı. Rasch sekmesinde a=1 sabittir.')
    if model in ('3PL','4PL'):
        estimation=st.radio('Kestirim',['MAP','ML'],key=f'python_mirt_estimation_{model}',horizontal=True)
        st.caption('MAP: kütüphane varsayılan önselleri; mevcut bilimsel kullanım sınırları geçerlidir.')
    if model=='NRM':
        raw_labels=st.text_input('Nominal kategoriler (kod sırası; başarı sırası değil)',value='A,B,C,D',key='python_mirt_nominal_labels')
        labels=tuple(v.strip().upper() for v in raw_labels.split(','));categories=len(labels)
    elif model not in BINARY:
        numeric=data[cols].apply(pd.to_numeric,errors='coerce') if cols else pd.DataFrame()
        maximum=numeric.max().max() if not numeric.empty else 2
        inferred=int(maximum)+1 if pd.notna(maximum) and np.isfinite(maximum) else 3
        categories=int(st.number_input('Kategori sayısı',min_value=3,max_value=20,value=max(3,min(20,inferred)),step=1,key=f'python_mirt_categories_{model}'))
    try:
        if len(cols)<3:raise ValueError('En az üç madde seçin.')
        if model in BINARY:
            x=data[cols].apply(pd.to_numeric,errors='raise')
            if not np.isfinite(x.to_numpy(float)).all() or not x.isin([0,1]).all().all():raise ValueError('Eksiksiz 0/1 yanıtlar gerekli; otomatik atama yapılmadı.')
        else:
            x,observed=prepare_polytomous(data[cols],model,categories=labels if model=='NRM' else None)
            if len(observed)!=categories:raise ValueError('Seçilen kategori sayısı verinin 0..C−1 kodlarıyla uyuşmuyor.')
            labels=observed
    except Exception as e:
        st.session_state.pop(f"python_mirt_fit_{model}", None)
        st.error(f'Veri ön kontrolü: {e}');return
    key=(tuple(cols),_data_fingerprint(x),categories,tuple(labels),estimation,'nine-model-report-v1')
    saved=st.session_state.get(f'python_mirt_fit_{model}')
    if saved is not None and (not isinstance(saved,dict) or saved.get('key')!=key):
        st.session_state.pop(f"python_mirt_fit_{model}", None)
        st.info('Veri veya analiz ayarları değişti; eski sonuç temizlendi. Yeniden hesaplayın.')
        saved=None
    if st.button('Python mirt ile hesapla',key=f'python_mirt_start_{model}'):
        st.session_state.pop(f"python_mirt_fit_{model}", None)
        try:
            with st.spinner('Python mirt kestirimi çalışıyor'):
                outcome=fit_python_mirt(x.to_numpy(),model='1PL' if model=='Rasch' else model,
                    n_categories=categories,estimation=estimation,common_discrimination=model=='1PL')
            saved=dict(key=key,outcome=outcome)
            st.session_state[f'python_mirt_fit_{model}']=saved
        except Exception as e:
            saved=None;st.error(f'Kestirim yapılamadı: {e}')
    if saved is None:return
    theta=np.linspace(-4,4,161)
    report=build_report(saved['outcome'],model=model,items=cols,labels=labels,theta=theta)
    if report.metadata['Yakınsama'] is True:st.success('Kestirim yakınsadı; bilimsel sertifika anlamına gelmez.')
    elif report.metadata['Yakınsama'] is False:st.error('Kestirim yakınsamadı. Gösterilen değerler geçicidir; başarılı analiz sayılmaz.')
    else:st.error('Yakınsama durumu doğrulanamadı; sonuç kesinleştirilemez.')
    for warning in report.warnings:st.warning(warning)
    for error in report.errors:st.error(error)
    summary=pd.DataFrame([report.metadata])
    st.markdown('### Model özeti');st.dataframe(summary,hide_index=True)
    st.download_button('Model özeti — CSV',_csv(summary),f'{model}_summary.csv','text/csv',key=f'python_mirt_summary_csv_{model}',on_click='ignore')
    if not report.parameters.empty:
        st.markdown('### Madde parametreleri');st.dataframe(safe_frame(report.parameters),hide_index=True)
        st.download_button('Madde parametreleri — CSV',_csv(report.parameters),f'{model}_parameters.csv','text/csv',key=f'python_mirt_parameters_csv_{model}',on_click='ignore')
        st.caption('a: ayırt edicilik; b: güçlük/eşik/adım. c/d: alt/üst asimptot. NRM A/D: kategori eğimi/kesişimi; kategori0 referans. SH/GA bu ortak tabloda üretilmez.')
    if report.curves.empty:return
    import plotly.graph_objects as go
    selected=st.selectbox('Grafik maddesi',cols,key=f'python_mirt_plot_item_{model}')
    shown=report.curves[report.curves.Madde==str(selected)]
    if model in BINARY:shown=shown[shown.Kategori=='1']
    figure=go.Figure()
    for category,group in shown.groupby('Kategori',sort=False):figure.add_scatter(x=group.Theta,y=group.Olasılık,name=f'Kategori {category}',mode='lines')
    figure.update_layout(title='ICC — Madde tepki olasılığı' if model in BINARY else 'Kategori tepki eğrileri',xaxis_title='θ',yaxis_title='Olasılık')
    st.plotly_chart(figure,use_container_width=True)
    st.download_button('Tüm tepki eğrileri — CSV',_csv(report.curves),f'{model}_curves.csv','text/csv',key=f'python_mirt_curves_csv_{model}',on_click='ignore')
    if report.information.empty:return
    info=report.information
    for title,col in [('Madde bilgi fonksiyonu',f'I_{selected}'),('Test bilgi fonksiyonu','Test bilgisi'),('Koşullu standart hata','SEM(θ)')]:
        figure=go.Figure();figure.add_scatter(x=info['θ'],y=info[col],mode='lines',name=title)
        figure.update_layout(title=title,xaxis_title='θ',yaxis_title=col);st.plotly_chart(figure,use_container_width=True)
    st.caption('SEM(θ)=1/√I_test(θ), parametre standart hatası veya kişi posterior SS değildir.')
    st.download_button('Madde/test bilgisi ve SEM — CSV',_csv(info),f'{model}_information.csv','text/csv',key=f'python_mirt_information_csv_{model}',on_click='ignore')
