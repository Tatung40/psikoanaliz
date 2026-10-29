"""Nine explicit Python mirt model routes sharing verified result reporting."""
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from core.irt_python_mirt_adapter import fit_python_mirt
from core.irt_python_mirt_report import MODELS,BINARY,build_report
from core.irt_b14 import prepare_polytomous
from reports.irt_b14_report import safe_frame
from core.irt_person_abilities import person_abilities
from ui.irt_solver_controls import solver_controls
from core.irt_rasch_wright import rasch_wright_tables,wright_figure,wright_exports


def _data_fingerprint(frame):
    """Order-sensitive digest of actual data, columns, index and dtype."""
    hashes=pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes()
    info=repr((tuple(frame.columns),tuple(map(str,frame.dtypes)),frame.shape)).encode('utf-8')
    return hashlib.sha256(info+hashes).hexdigest()


def _csv(frame):return safe_frame(frame).to_csv(index=False).encode('utf-8-sig')


def render(data,items=(),id_col=None,group_col=None,model='1PL'):
    if model not in MODELS:st.error('Desteklenmeyen model');return
    st.subheader(f'Python mirt Kütüphanesi — {model}')
    st.caption('Python mirt 1.2.0 | Sonuç ekranı sürümü: nine-model-report-v2 | Bağımsız bilimsel sertifika veya formel DIF değildir.')
    choices=[c for c in data.columns if c not in (id_col,group_col)]
    cols=st.multiselect('Madde sütunları',choices,default=[c for c in items if c in choices] or choices[:10],key=f'python_mirt_cols_{model}')
    # Display-only controls: never alter calibration or claim unsupported solver settings.
    with st.expander('Çıktılar ve raporlama — SPSS tarzı seçimler',expanded=False):
        st.caption('Varsayılan: tüm mevcut çıktılar. Seçimler yalnızca görünümü ve dışa aktarımı etkiler; model kalibrasyonunu değiştirmez.')
        output_names={
            'summary':'Model özeti, LogLik, AIC/BIC ve yakınsama',
            'parameters':'Madde parametreleri',
            'persons':'Birey EAP yetenekleri ve posterior SH',
            'curves':'Madde/kategori tepki eğrileri',
            'information':'Madde ve test bilgisi, koşullu SEM',
            'comparison':'Aynı veri üzerinde model karşılaştırması (ML AIC/BIC)',
            'itemfit':'Madde uyumu: Infit, Outfit, S-X² (kütüphane API)',
        }
        outputs={name:st.checkbox(label,value=not (model=='NRM' and name=='itemfit'),disabled=model=='NRM' and name=='itemfit',key=f'python_mirt_output_{model}_{name}')
                 for name,label in output_names.items()}
    solver_options=solver_controls(f"python_mirt_solver_{model}",advanced_key=f"python_mirt_solver_advanced_{model}")
    if model=='NRM':st.caption('Nominal kategoriler için ordinal Infit/Outfit/S-X² sunulmaz.')
    with st.expander('Grafik ve çıktı ayrıntıları',expanded=False):
        st.caption('Varsayılan seçimler mevcut raporun bütün grafiklerini korur. Seçimler hesaplamayı değil yalnızca sunumu etkiler.')
        chart_options={
            'item_info':'Madde bilgi fonksiyonu (IIF)',
            'test_info':'Test bilgi fonksiyonu (TIF)',
            'sem':'Koşullu ölçme standart hatası SEM(θ)',
        }
        charts={name:st.checkbox(label,value=True,key=f'python_mirt_chart_{model}_{name}')
                for name,label in chart_options.items()}
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
        minimum=numeric.min().min() if not numeric.empty else 0
        origin=1 if minimum==1 else 0
        inferred=int(maximum)-origin+1 if pd.notna(maximum) and np.isfinite(maximum) else 3
        categories=int(st.number_input('Kategori sayısı',min_value=3,max_value=20,value=max(3,min(20,inferred)),step=1,key=f'python_mirt_categories_{model}'))
    try:
        if len(cols)<3:raise ValueError('En az üç madde seçin.')
        if model in BINARY:
            x=data[cols].apply(pd.to_numeric,errors='raise')
            if not np.isfinite(x.to_numpy(float)).all() or not x.isin([0,1]).all().all():raise ValueError('Eksiksiz 0/1 yanıtlar gerekli; otomatik atama yapılmadı.')
        else:
            x,observed=prepare_polytomous(data[cols],model,categories=labels if model=='NRM' else None)
            if len(observed)!=categories:raise ValueError('Seçilen kategori sayısı gözlenen etiketlerle uyuşmuyor.')
            labels=observed
    except Exception as e:
        st.session_state.pop(f"python_mirt_fit_{model}", None)
        st.error(f'Veri ön kontrolü: {e}');return
    identity_hash=_data_fingerprint(data[[id_col]]) if id_col in data.columns else None
    key=(identity_hash,tuple(cols),_data_fingerprint(x),categories,tuple(labels),estimation,tuple(sorted(solver_options.items())),'nine-model-report-v2')
    saved=st.session_state.get(f'python_mirt_fit_{model}')
    if saved is not None and (not isinstance(saved,dict) or saved.get('key')!=key):
        st.session_state.pop(f"python_mirt_fit_{model}", None)
        st.info('Veri veya analiz ayarları değişti; eski sonuç temizlendi. Yeniden hesaplayın.')
        saved=None
    pool_key=(_data_fingerprint(x),tuple(cols),tuple(labels),categories,tuple(sorted(solver_options.items())))
    if st.button('Python mirt ile hesapla',key=f'python_mirt_start_{model}'):
        st.session_state.pop(f"python_mirt_fit_{model}", None)
        st.session_state.setdefault('python_mirt_comparison_pool',{}).pop((pool_key,model),None)
        try:
            with st.spinner('Python mirt kestirimi çalışıyor'):
                fit_kwargs=dict(model=model,n_categories=categories,estimation=estimation,
                                common_discrimination=model=='1PL')
                fit_kwargs.update(solver_options)
                outcome=fit_python_mirt(x.to_numpy(),**fit_kwargs)
            saved=dict(key=key,outcome=outcome)
            st.session_state[f'python_mirt_fit_{model}']=saved
        except Exception as e:
            saved=None;st.error(f'Kestirim yapılamadı: {e}')
    if saved is None:return
    # A comparison pool is populated only with actual converged ML fits on identical
    # item data and identical category coding. MAP is not compared by ordinary AIC/BIC.
    pool=st.session_state.setdefault('python_mirt_comparison_pool',{})
    if estimation=='ML':
        try:
            from core.irt_library_diagnostics import library_fit_diagnostics
            diag=library_fit_diagnostics(saved['outcome'].fit)
            if diag['converged'] and all(np.isfinite(float(diag[z])) for z in ('log_likelihood','aic','bic')):
                pool[(pool_key,model)]={'Model':model,'N':len(x),'Madde':len(cols),
                    'LogLik':float(diag['log_likelihood']),'AIC':float(diag['aic']),
                    'BIC':float(diag['bic']),'Parametre':int(diag['n_parameters'])}
        except (ValueError,TypeError,KeyError,AttributeError):
            pool.pop((pool_key,model),None)
    if outputs['comparison']:
        st.markdown('### Python model karşılaştırması')
        st.caption('Yalnızca bu oturumda aynı madde verisi, kategori kodlaması ve ML yöntemiyle yakınsamış modeller karşılaştırılır. MAP için klasik AIC/BIC karşılaştırması sunulmaz. Düşük AIC/BIC göreli uyumu gösterir, bilimsel geçerlik kanıtı değildir.')
        compatible=[v for (k,_),v in pool.items() if k==pool_key]
        if estimation!='ML':
            st.info('MAP analizi klasik ML AIC/BIC karşılaştırmasına dahil edilmez.')
        elif len(compatible)<2:
            st.info('Karşılaştırma için aynı veriyle en az iki ML modeli çalıştırın.')
        else:
            comparison=pd.DataFrame(compatible).sort_values(['AIC','Model']).reset_index(drop=True)
            comparison['ΔAIC']=comparison['AIC']-comparison['AIC'].min()
            comparison['ΔBIC']=comparison['BIC']-comparison['BIC'].min()
            st.dataframe(comparison,hide_index=True)
            st.download_button('Model karşılaştırması — CSV',_csv(comparison),'model_comparison.csv',
                               'text/csv',key=f'python_mirt_compare_csv_{model}',on_click='ignore')
    theta=np.linspace(-4,4,161)
    report=build_report(saved['outcome'],model=model,items=cols,labels=labels,theta=theta)
    if report.metadata['Yakınsama'] is True:st.success('Kestirim yakınsadı; bilimsel sertifika anlamına gelmez.')
    elif report.metadata['Yakınsama'] is False:st.error('Kestirim yakınsamadı. Gösterilen değerler geçicidir; başarılı analiz sayılmaz.')
    else:st.error('Yakınsama durumu doğrulanamadı; sonuç kesinleştirilemez.')
    for warning in report.warnings:st.warning(warning)
    for error in report.errors:st.error(error)
    if outputs['itemfit'] and model!='NRM':
        st.markdown('### Madde uyumu — Python mirt itemfit')
        st.caption('Infit ve Outfit MNSQ, S-X²: Python mirt 1.2.0 kamuya açık itemfit API. Yalnızca yakınsamış modelde gösterilir. Model ailesine göre istatistik desteği değişebilir; desteklenmeyen sonuçlar uydurulmaz.')
        if report.metadata.get('Yakınsama') is not True or report.errors:
            st.warning('Yakınsama veya rapor hatası nedeniyle madde uyumu hesaplanmadı.')
        else:
            try:
                import mirt
                fit_table=mirt.itemfit(saved['outcome'].fit,x.to_numpy(dtype=int),
                                       statistics=['infit','outfit','S_X2'])
                fit_frame=safe_frame(fit_table)
                if fit_frame.empty:
                    st.info('Kütüphane bu model için madde uyum tablosu döndürmedi.')
                else:
                    st.dataframe(fit_frame,hide_index=True,use_container_width=True)
                    st.download_button('Madde uyumu — CSV',_csv(fit_frame),f'{model}_itemfit.csv',
                                       'text/csv',key=f'python_mirt_itemfit_csv_{model}',on_click='ignore')
            except (AttributeError,ValueError,TypeError,KeyError,ImportError,RuntimeError) as exc:
                st.warning(f'Kütüphane madde uyumu hesaplayamadı: {exc}')
    rasch_tables=None
    summary=pd.DataFrame([report.metadata])
    if outputs['summary']:
        st.markdown('### Model özeti');st.dataframe(summary,hide_index=True)
    if outputs['summary']:st.download_button('Model özeti — CSV',_csv(summary),f'{model}_summary.csv','text/csv',key=f'python_mirt_summary_csv_{model}',on_click='ignore')
    if outputs['parameters'] and not report.parameters.empty:
        st.markdown('### Madde parametreleri');st.dataframe(safe_frame(report.parameters),hide_index=True)
        st.download_button('Madde parametreleri — CSV',_csv(report.parameters),f'{model}_parameters.csv','text/csv',key=f'python_mirt_parameters_csv_{model}',on_click='ignore')
        st.caption('a: ayırt edicilik; b: güçlük/eşik/adım. c/d: alt/üst asimptot. NRM A/D: kategori eğimi/kesişimi; kategori0 referans. SH/GA bu ortak tabloda üretilmez.')
    if outputs['persons']:st.markdown('### Birey Yetenek Kestirimleri (θ)')
    if outputs['persons']:st.caption('N(0,1) EAP; SH(θ) posterior SS’dir, bilgi SEM’i değildir. '+('Rasch: desteklenen mirt.fscores GH61 API.' if model=='Rasch' else '161 noktalı [-4,4] ızgara.'))
    if outputs['persons'] and (report.metadata.get('Yakınsama') is not True or report.errors):
        st.warning('Yakınsama veya rapor hatası nedeniyle kişi kestirimleri kesin sonuç olarak sunulmaz.')
    elif outputs['persons'] and not report.curves.empty:
        try:
            ids=data[id_col].to_numpy(copy=True) if id_col in data.columns else None
            if model=='Rasch':
                rasch_tables=rasch_wright_tables(saved['outcome'],x,item_names=cols,identifiers=ids)
                raw_people=rasch_tables['persons']
                people=pd.DataFrame({'Birey':raw_people.Birey,'Ham puan':x.to_numpy().sum(1),
                    'Yetenek (θ)':raw_people['θ'],'SH (θ)':raw_people['Posterior SD'],'Yöntem':'EAP','Model':model})
            else:people=person_abilities(x.to_numpy(),report.curves,model=model,identifiers=ids)
            st.dataframe(safe_frame(people).round(4),hide_index=True,use_container_width=True)
            st.download_button('Birey yetenekleri — CSV',_csv(people),
                               f'{model}_person_theta.csv','text/csv',
                               key=f'python_mirt_persons_csv_{model}',on_click='ignore')
            import io
            output=io.BytesIO()
            with pd.ExcelWriter(output,engine='openpyxl') as writer:
                safe_frame(people).to_excel(writer,index=False,sheet_name='Birey Yetenekleri')
            st.download_button('Birey yetenekleri — Excel',output.getvalue(),
                               f'{model}_person_theta.xlsx',
                               'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                               key=f'python_mirt_persons_xlsx_{model}',on_click='ignore')
        except (ValueError,TypeError,KeyError,IndexError) as exc:
            st.error(f'Birey yetenek kestirimleri hesaplanamadı: {exc}')
    # Canonical verified Phase2 Wright data; preserve PR73's opt-in UI gate. Conservative scientific gate: Rasch only;
    # other families remain disabled pending model-specific independent validation.
    if model=='Rasch' and report.metadata.get('Yakınsama') is True and not report.errors:
        with st.expander('Wright Map — Rasch (Python mirt)',expanded=False):
            if st.checkbox('Kütüphane Wright Map grafiğini göster',value=False,key='python_mirt_wright_Rasch'):
                try:
                    if rasch_tables is None:
                        rasch_tables=rasch_wright_tables(saved['outcome'],x,item_names=cols,
                            identifiers=data[id_col].to_numpy(copy=True) if id_col in data.columns else None)
                    st.plotly_chart(wright_figure(rasch_tables),use_container_width=True)
                    csv,excel=wright_exports(rasch_tables)
                    for name in ('items','histogram'):st.dataframe(safe_frame(rasch_tables[name]),hide_index=True)
                    for name,blob in csv.items():
                        st.download_button(f'Wright {name} — CSV',blob,f'Rasch_wright_{name}.csv','text/csv',key=f'rasch_wright_{name}',on_click='ignore')
                    st.download_button('Wright tabloları — Excel',excel,'Rasch_wright.xlsx',key='rasch_wright_excel',on_click='ignore')
                except (AttributeError,ValueError,TypeError,ImportError,RuntimeError) as exc:
                    st.warning(f'Python mirt Wright Map üretilemedi: {exc}')
    elif model!='Rasch':
        st.caption('Wright Map: bu model için bağımsız bilimsel doğrulama kapısı henüz açılmadı.')
    if model in BINARY and report.metadata.get('Yakınsama') is True and not report.errors:
        _binary_extras(saved,report,data,x,cols,id_col,model,rasch_tables)
    if report.curves.empty:return
    if not outputs['curves'] and not outputs['information']:return
    # Render server-side PNG charts: avoids mobile Safari dynamic Plotly module imports.
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    selected=st.selectbox('Grafik maddesi',cols,key=f'python_mirt_plot_item_{model}')
    shown=report.curves[report.curves.Madde==str(selected)]
    if model in BINARY:shown=shown[shown.Kategori=='1']
    if outputs['curves']:
        fig,ax=plt.subplots(figsize=(7,4))
        for category,group in shown.groupby('Kategori',sort=False):
            ax.plot(group['Theta'],group['Olasılık'],label=f'Kategori {category}')
        ax.set(title='ICC — Madde tepki olasılığı' if model in BINARY else 'Kategori tepki eğrileri',
               xlabel='Theta',ylabel='Olasılık',ylim=(0,1))
        ax.grid(alpha=.2)
        if len(shown['Kategori'].unique())>1:ax.legend()
        fig.tight_layout()
        try:st.pyplot(fig,use_container_width=True)
        finally:plt.close(fig)
        st.download_button('Tüm tepki eğrileri — CSV',_csv(report.curves),f'{model}_curves.csv','text/csv',key=f'python_mirt_curves_csv_{model}',on_click='ignore')
    if not outputs['information'] or report.information.empty:return
    info=report.information
    for chart_key,title,col in [('item_info','Madde bilgi fonksiyonu',f'I_{selected}'),('test_info','Test bilgi fonksiyonu','Test bilgisi'),('sem','Koşullu standart hata','SEM(θ)')]:
        if not charts[chart_key]:continue
        fig,ax=plt.subplots(figsize=(7,4))
        ax.plot(info['θ'],info[col])
        ax.set(title=title,xlabel='Theta',ylabel=col)
        ax.grid(alpha=.2)
        fig.tight_layout()
        try:st.pyplot(fig,use_container_width=True)
        finally:plt.close(fig)
    st.caption('SEM(θ)=1/√I_test(θ), parametre standart hatası veya kişi posterior SS değildir.')
    st.download_button('Madde/test bilgisi ve SEM — CSV',_csv(info),f'{model}_information.csv','text/csv',key=f'python_mirt_information_csv_{model}',on_click='ignore')


def _binary_extras(saved,report,data,x,cols,id_col,model,rasch_tables):
    """B13 output types on the library route: parameter SE/CI, eight charts as 300 dpi PNG, report bundle."""
    from core.irt_binary_inference import parameter_inference,STATUS
    from reports.irt_binary_report import kinds,png_bytes,bundle
    with st.expander('Ek bilimsel çıktılar — parametre SH/GA, grafikler (300 dpi PNG) ve rapor paketi',expanded=False):
        if not st.checkbox('Ek çıktıları göster',value=False,key=f'python_mirt_extras_{model}'):return
        try:
            ids=data[id_col].to_numpy(copy=True) if id_col in data.columns else None
            if model=='Rasch':
                tables=rasch_tables or rasch_wright_tables(saved['outcome'],x,item_names=cols,identifiers=ids)
                persons=pd.DataFrame({'Birey':tables['persons'].Birey,'Ham puan':x.to_numpy().sum(1),
                    'Yetenek (θ)':tables['persons']['θ'],'SH (θ)':tables['persons']['Posterior SD']})
            else:persons=person_abilities(x.to_numpy(),report.curves,model=model,identifiers=ids)
            inference,reason=parameter_inference(saved['outcome'].fit,model,cols)
        except (AttributeError,ValueError,TypeError,KeyError,ImportError,RuntimeError) as exc:
            st.warning(f'Ek çıktılar üretilemedi: {exc}');return
        st.markdown('#### Madde parametresi SH ve %95 Wald güven aralığı')
        if inference is None:st.info(reason)
        else:
            st.dataframe(safe_frame(inference).round(4),hide_index=True,use_container_width=True)
            st.download_button('Parametre SH/GA — CSV',_csv(inference),f'{model}_parameter_se_ci.csv','text/csv',
                               key=f'python_mirt_se_csv_{model}',on_click='ignore')
        st.caption(STATUS)
        kind=st.selectbox('Grafik',kinds(model),key=f'python_mirt_extra_kind_{model}')
        chosen=None
        if kind in ('ICC','IIC'):
            chosen=st.multiselect('Maddeler',cols,default=cols[:8],key=f'python_mirt_extra_items_{model}') or cols[:1]
        image=png_bytes(report,persons,kind,chosen)
        st.image(image,use_container_width=True)
        st.download_button('PNG indir (300 dpi)',image,f'{model}_{kind}.png','image/png',
                           key=f'python_mirt_extra_png_{model}',on_click='ignore')
        if kind=='Wright haritası' and model not in ('Rasch','1PL','2PL'):
            st.caption('3PL/4PL: b ortak θ ölçeğindedir ancak P = .5 noktası değildir; harita betimseldir.')
        store=st.session_state.setdefault('python_mirt_bundle',{})
        if st.button('Rapor paketini hazırla (Excel + ZIP)',key=f'python_mirt_bundle_run_{model}'):
            itemfit=None
            try:
                import mirt
                itemfit=safe_frame(mirt.itemfit(saved['outcome'].fit,x.to_numpy(dtype=int),statistics=['infit','outfit','S_X2']))
            except (AttributeError,ValueError,TypeError,KeyError,ImportError,RuntimeError):
                itemfit=None
            store[model]={'key':saved['key'],'files':bundle(report,persons,inference,reason,itemfit)}
        entry=store.get(model)
        if entry and entry['key']==saved['key']:
            xlsx,archive=entry['files']
            st.download_button('Excel — tüm tablolar',xlsx,f'{model}_rapor.xlsx',
                               'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                               key=f'python_mirt_bundle_xlsx_{model}',on_click='ignore')
            st.download_button('ZIP — rapor paketi',archive,f'{model}_rapor.zip','application/zip',
                               key=f'python_mirt_bundle_zip_{model}',on_click='ignore')
