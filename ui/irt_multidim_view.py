"""Fail-closed Streamlit route for the existing public multidimensional library."""
import io
import numpy as np
import pandas as pd
import streamlit as st
from core.reproducible import displayed
from core.irt_multidim import BIFACTOR_RECOVERY,MODELS,LIMITATION,fit,report,structure,support_matrix
from core.python_mirt_dif import fingerprint
from core import irt_multidim_factorized as factorized
from core import irt_multidim_rsm_stack as rsm
from core import irt_multidim_general_report as general
from reports.irt_multidim_report import downloads,figure,export_fingerprint
from core.irt_mirt_policy import profiles,evidence_fingerprint
from core.irt_mirt_production import decorate_native,refresh_policy
from time import perf_counter

KEY='mirt_multidim_result'


def render(data=None,item_cols=(),id_col=None):
 st.subheader('🧬 Çok Boyutlu MTK (MIRT)')
 with st.expander('API ve bağımsız doğrulama kapsamı'):
  supported=support_matrix().copy()
  ready={v['model'] for v in profiles().values()}
  supported.loc[supported.Model.isin(ready),'Status']='SCOPED NUMERICAL PROFILES AVAILABLE / inference WITHHELD'
  st.dataframe(supported,hide_index=True)
  st.caption('Kayıtlı 2D/3D 2PL/GRM karşılaştırmaları kayıtlı kapsamda sayısal uygulama doğrulamasıdır; çıkarımsal sertifikasyon değildir. Native profiller validation/mirt_stage1; ayrı SciPy profil/kısıt/engel matrisi validation/mirt_phase3 raporundadır.')
 upload=st.file_uploader('MIRT verisi (CSV/XLSX)',type=['csv','xlsx'],key='md_file')
 if upload:
  try:data=pd.read_excel(io.BytesIO(upload.getvalue())) if upload.name.endswith('.xlsx') else pd.read_csv(io.BytesIO(upload.getvalue()))
  except Exception as exc:st.session_state.pop(KEY,None);st.error(f'Veri okunamadı: {exc}');return
 if data is None or data.empty:st.session_state.pop(KEY,None);st.info('Veri yükleyin.');return
 if not data.columns.is_unique or any(not isinstance(c,str) for c in data):
  st.session_state.pop(KEY,None);st.error('Benzersiz metin sütun adları gerekli');return
 id_options=['(satır indeksi)']+list(data.columns)
 person=st.selectbox('Kişi kimliği',id_options,index=id_options.index(id_col) if id_col in data else 0,key='md_id')
 choices=[c for c in data if c!=person]
 items=st.multiselect('Maddeler',choices,default=[c for c in item_cols if c in choices] or choices,key='md_items')
 model=st.selectbox('MIRT modeli',MODELS+factorized.MODELS+rsm.MODELS+general.MODELS,index=MODELS.index('2PL'),key='md_model')
 current_profiles=profiles(model) if model in ('2PL','GRM','GPCM','PCM','Rasch') else {}
 # Preserve declarative input specs on evidence withdrawal; only the certificate is refreshed, never the fit.
 known=st.session_state.setdefault('md_profile_specs',{})
 known.update(current_profiles)
 available={name:spec for name,spec in known.items() if spec['model']==model}
 preset=st.selectbox('Kayıtlı CFA doğrulama profili (veri seçmez)', ['Özel yapı / deneysel']+list(available),key='md_profile') if available else 'Özel yapı / deneysel'
 cfg=available.get(preset)
 if cfg:st.info('Profil yalnız gözlenebilir yapı/ayarları uygular. Gizli dağılım varsayımları kullanıcı verisinden doğrulanamaz.')
 restricted=model in factorized.MODELS;stacked=model in rsm.MODELS;advanced=model in general.MODELS;c=.2;u=1.
 st.warning(general.LIMITATION if advanced else LIMITATION)
 if stacked:st.warning(rsm.LIMITATION)
 if restricted or advanced:
  if restricted:st.warning(factorized.LIMITATION)
  c=st.number_input('Başlangıç alt asimptot c (kestirilir)' if advanced and general.CONFIG[model][1] else 'Sabit alt asimptot c (kestirilmez)',min_value=0.,max_value=.5,value=.2,step=.01,key='md_fixed_c')
  if model.startswith('4PL'):u=st.number_input('Başlangıç üst asimptot d (kestirilir)' if advanced and general.CONFIG[model][2] else 'Sabit üst asimptot d (kestirilmez)',min_value=.5,max_value=1.,value=.95,step=.01,key='md_fixed_u')
 dims=st.selectbox('Boyut sayısı',[cfg['q']] if cfg else general.allowed_dimensions(model) if advanced else [2,3,4,5],key='md_dims')
 mode=st.selectbox('Yapı',['Doğrulayıcı'] if cfg else ['Doğrulayıcı','Keşfedici (pozitif, üçgen)'],key='md_mode')
 efa=mode.startswith('Keşfedici')
 corr_options=[{'full':'Tümü',True:'Yalnız F1–F2',False:'Ortogonal'}[cfg['correlated']]] if cfg else ['Tümü'] if advanced else ['Ortogonal','Yalnız F1–F2','Tümü']
 corr=st.selectbox('Boyut korelasyonları',corr_options,key='md_corr')
 correlated={'Ortogonal':False,'Yalnız F1–F2':True,'Tümü':'full'}[corr]
 pattern=np.zeros((len(items),dims),int)
 if len(items):pattern[np.arange(len(items)),np.minimum(np.arange(len(items))*dims//len(items),dims-1)]=1
 if cfg and len(items)==cfg['k']:pattern=np.asarray(cfg['pattern'],int)
 if efa:
  pattern[:,:]=1;pattern[:min(dims,len(items)),:]=np.tril(np.ones((dims,dims),int))[:len(items),:]
 if model=='Bifactor':
  groups=st.text_input('Özgül gruplar (madde sırasıyla 0,1,2…)',','.join(str(i*max(dims-1,1)//max(len(items),1)) for i in range(len(items))),key=f'md_groups_{dims}')
  chosen=groups
  st.caption(BIFACTOR_RECOVERY)
 else:
  matrix=st.data_editor(pd.DataFrame(pattern,index=items,columns=[f'F{j+1}' for j in range(dims)]),key=f'md_profile_pattern_{preset}' if cfg else f'md_pattern_{len(items)}_{dims}_{efa}',num_rows='fixed',disabled=bool(cfg))
  chosen=matrix.to_numpy().tolist()
 labels=None
 if model=='NRM':
  observed=list(dict.fromkeys(str(v).strip().upper() for v in data[items].to_numpy().ravel())) if items else []
  labels=tuple(v.strip().upper() for v in st.text_input('Nominal etiketler: kod sırası, başarı sırası değil',','.join(observed),key='md_labels').split(','))
 nodes=st.selectbox('Eksen başına GH düğümü',[cfg['n_quadpts']] if cfg else [21,41,61] if restricted or stacked else {2:[9,15,21],3:[5,9,15],4:[5,7],5:[5]}[dims],index=({2:1,3:2}[dims] if advanced else 0),key=f'md_general_quad_{dims}' if advanced else f'md_quad_{dims}')
 iterations=st.number_input('Maksimum iterasyon',min_value=1,max_value=3000,value=cfg['max_iter'] if cfg else 2000 if advanced else 1000,step=100,key='md_profile_iter' if cfg else 'md_general_iter' if advanced else 'md_iter',disabled=bool(cfg))
 tolerance=st.number_input('Göreli amaç toleransı (SciPy)' if advanced else 'EM toleransı',min_value=1e-15 if advanced else 1e-10,max_value=1e-3,value=cfg['tol'] if cfg else 1e-13 if advanced else 1e-7,format='%.10g',key='md_profile_tol' if cfg else 'md_general_tol' if advanced else 'md_tol',disabled=bool(cfg))
 settings=dict(c=c,u=u,items=items,person=person,model=model,dims=dims,efa=efa,correlated=correlated,pattern=chosen,labels=labels,n_quadpts=nodes,max_iter=int(iterations),tol=tolerance)
 key=fingerprint(data,settings);saved=st.session_state.get(KEY)
 if saved and saved['key']!=key:st.session_state.pop(KEY,None);saved=None
 try:
  if not items:raise ValueError('Maddeleri seçin')
  if cfg and len(items)!=cfg['k']:raise ValueError(f"Kayıtlı profil {cfg['k']} madde gerektirir; özel yapı deneysel olarak kullanılabilir.")
  ids=data.index.tolist() if person==id_options[0] else data[person].tolist()
  if pd.Series(ids).isna().any() or not pd.Index(ids).is_unique:raise ValueError('Kimlikler eksiksiz ve benzersiz olmalı')
  if model=='Bifactor':
   chosen=[int(v.strip()) for v in chosen.split(',')]
   if len(chosen)!=len(items) or len(set(chosen))+1!=dims or efa or correlated:raise ValueError('Bifactor: ortogonal, boyut=özgül grup+1; grup başına ≥3 madde')
  elif restricted:factorized.validate(data[items],chosen,correlated,efa,c,u)
  elif stacked:rsm.validate(data[items],chosen,correlated,efa)
  elif advanced:general.specification(data[items],model,chosen,correlated,efa,c,u)
  else:structure(chosen,correlated,efa)
 except Exception as exc:st.session_state.pop(KEY,None);st.error(f'Ön kontrol: {exc}');return
 cost=f'{len(data)*dims} yapısal satır × {nodes} GH düğümü; TEK ortak adım kestirimi' if stacked else f'{dims} bağımsız blok × {nodes} GH düğümü' if restricted else f'{nodes**dims} GH düğümü'
 st.caption(f'{len(data)} kişi × {len(items)} madde; {cost}. Eksik veri otomatik doldurulmaz. İşlem bütçesi aşılırsa kestirim engellenir.')
 if st.button('MIRT kestir',key='md_run',type='primary'):
  st.session_state.pop(KEY,None);saved=None
  try:
   started=perf_counter();outcome=None
   with st.spinner('Deneysel SciPy genel CFA marjinal ML kestirimi' if advanced else 'Gerçek Python mirt marjinal ML kestirimi'):
    if restricted:result=factorized.report(factorized.fit(data[items],model,chosen,correlated=correlated,exploratory=efa,c=c,u=u,n_quadpts=nodes,max_iter=int(iterations),tol=tolerance),ids)
    elif advanced:result=general.fit_report(data[items],model,chosen,correlated=correlated,exploratory=efa,c=c,u=u,n_quadpts=nodes,max_iter=int(iterations),tol=tolerance,person_ids=ids)
    elif stacked:result=rsm.report(rsm.fit(data[items],model,chosen,correlated=correlated,exploratory=efa,n_quadpts=nodes,max_iter=int(iterations),tol=tolerance),ids)
    else:
     outcome=fit(data[items],model,chosen,correlated=correlated,n_quadpts=nodes,max_iter=int(iterations),tol=tolerance,labels=labels,exploratory=efa)
     fitted_at=perf_counter();base_result=report(outcome,ids)
     result=decorate_native(outcome,base_result,ids,extra_metadata=dict(Input_fingerprint=key,ID_column=person,
                              Calibration_seconds=fitted_at-started,Base_reporting_seconds=perf_counter()-fitted_at))
   if outcome is None or model not in ('2PL','GRM','GPCM','PCM','Rasch','1PL (a=1)','NRM') or efa:
    result['metadata']['Input_fingerprint']=key;result['metadata']['ID_column']=person
   saved=dict(key=key,result=result,outcome=outcome,policy_proof=evidence_fingerprint());st.session_state[KEY]=saved
  except Exception as exc:st.error(f'Kestirim/raporlama başarısız: {exc}')
 if not saved:return
 if saved.get('outcome') is not None and saved.get('policy_proof')!=evidence_fingerprint():
  saved['result']=refresh_policy(saved['outcome'],saved['result']);saved['policy_proof']=evidence_fingerprint()
  saved.pop('bundle_key',None)
 result=saved['result'];tables=result['tables'];displayed(result)
 st.info(str(result['metadata'].get('Status','EXPERIMENTAL / NOT CERTIFIED')))
 st.caption(str(result['metadata'].get('Limitations','')))
 if not result['metadata']['Converged']:st.error('NOT CONVERGED — yalnız geçici parametreler; puanlar ve eğriler kapalı.')
 if result['metadata'].get('At_box_bound'):st.warning('Kutu sınırında parametre var (Boundaries tablosu): ilgili yük/eşik ve standart hata yorumlanmamalı; örneklemi büyütün veya yapıyı gözden geçirin.')
 tabs=st.tabs(['Özet / yakınsama','Parametreler / yapı','Kişi puanları','Bilgi / SEM','Eğriler','Raporlar'])
 for tab,names in zip(tabs[:4],[['Summary','Fit','Output_scope'],['Parameters','SharedSteps','Pattern','Covariance','Correlation','Boundaries','LoadingMatrix'],['EAP','MAP','PosteriorCovariance'],['SEM','IIF','TIF','Directional_TIF','Directional_IIF','DerivedDiagnostics']]):
  with tab:
   for name in names:
    if name in tables:st.write(name);st.dataframe(tables[name],hide_index=True)
 with tabs[1]:
  if 'LoadingMatrix' in tables:
   import plotly.graph_objects as go
   t=tables['LoadingMatrix'];names=t.Item.astype(str)
   if 'Category' in t:names=names+' / '+t.Category.astype(str)
   m=t.filter(regex='^a[1-5]
 with tabs[4]:
  if 'Curves' in tables:
   item=st.selectbox('Eğri maddesi',items,key='md_plot_item');category=st.selectbox('Kategori',list(result['metadata']['Labels']),key='md_plot_cat')
   if 'Directional_Curves' in tables:
    import plotly.graph_objects as go
    trace=tables['Directional_Curves']
    direction=st.selectbox('Tepki eğrisi yönü',trace.Direction.unique().tolist(),key='md_plot_direction')
    p=trace[(trace.Item==item)&(trace.Direction==direction)]
    g=go.Figure()
    for name,part in p.groupby('Category'):g.add_scatter(x=part.Theta,y=part.Probability,name=str(name))
    st.plotly_chart(g.update_layout(title=f'{item}: θ=t e doğrultusunda kategori olasılıkları',xaxis_title='t',yaxis_title='Olasılık'),key='md_directional_response',use_container_width=True)
   fig=figure(result,item,category);st.pyplot(fig)
   import matplotlib.pyplot as plt
   plt.close(fig)
 with tabs[5]:
  bundle_key=export_fingerprint(result,item,category)
  if saved.get('bundle_key')!=bundle_key:saved['bundle']=downloads(result,item,category);saved['bundle_key']=bundle_key
  for ext,mime in [('csv','text/csv'),('xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),('png','image/png'),('zip','application/zip'),('report','text/markdown')]:
   if saved['bundle'][ext] is not None:st.download_button('MIRT — '+ext.upper(),saved['bundle'][ext],'mirt_results.'+('md' if ext=='report' else ext),mime,key='md_download_'+ext)
)
   st.plotly_chart(go.Figure(go.Heatmap(z=m.to_numpy(),x=m.columns,y=names)).update_layout(title='IRT ayırıcılık vektörleri — standartlaştırılmış FA yükleri değildir'),key='md_loading_heatmap',use_container_width=True)
 with tabs[2]:
  if 'EAP' in tables and {'F1','F2'}<=set(tables['EAP']):
   import plotly.graph_objects as go
   t=tables['EAP']
   st.plotly_chart(go.Figure(go.Scattergl(x=t.F1,y=t.F2,mode='markers',text=t.ID.astype(str))).update_layout(title='Koşullu EAP kişi puanları',xaxis_title='F1',yaxis_title='F2'),key='md_person_scatter',use_container_width=True)
 with tabs[3]:
  if 'Directional_TIF' in tables:
   import plotly.graph_objects as go
   d=tables['Directional_TIF'];cut=tables['SEM'].query('Theta2 == 0')[['Node','Theta1']]
   g=go.Figure()
   for name,part in d.merge(cut,on='Node').groupby('Direction'):
    part=part.sort_values('Theta1');g.add_scatter(x=part.Theta1,y=part.Information,name=str(name))
   st.plotly_chart(g.update_layout(title="Yönlü test bilgisi e′I(θ)e; diğer koordinatlar 0",xaxis_title='F1',yaxis_title='Bilgi'),key='md_directional_plot',use_container_width=True)
 item=items[0];category=result['metadata']['Labels'][0]
 with tabs[4]:
  if 'Curves' in tables:
   item=st.selectbox('Eğri maddesi',items,key='md_plot_item');category=st.selectbox('Kategori',list(result['metadata']['Labels']),key='md_plot_cat')
   fig=figure(result,item,category);st.pyplot(fig)
   import matplotlib.pyplot as plt
   plt.close(fig)
 with tabs[5]:
  bundle_key=export_fingerprint(result,item,category)
  if saved.get('bundle_key')!=bundle_key:saved['bundle']=downloads(result,item,category);saved['bundle_key']=bundle_key
  for ext,mime in [('csv','text/csv'),('xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),('png','image/png'),('zip','application/zip'),('report','text/markdown')]:
   if saved['bundle'][ext] is not None:st.download_button('MIRT — '+ext.upper(),saved['bundle'][ext],'mirt_results.'+('md' if ext=='report' else ext),mime,key='md_download_'+ext)
