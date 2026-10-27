"""Fail-closed Streamlit route for the existing public multidimensional library."""
import io
import numpy as np
import pandas as pd
import streamlit as st
from core.irt_multidim import MODELS,LIMITATION,fit,report,structure,support_matrix
from core.python_mirt_dif import fingerprint
from core import irt_multidim_factorized as factorized
from reports.irt_multidim_report import downloads,figure

KEY='mirt_multidim_result'


def render(data=None,item_cols=(),id_col=None):
 st.subheader('🧬 Çok Boyutlu MTK (MIRT)');st.warning(LIMITATION)
 with st.expander('API ve bağımsız doğrulama kapsamı'):
  st.dataframe(support_matrix(),hide_index=True)
  st.caption('Kayıtlı 2D/3D 2PL/GRM karşılaştırmaları sayısal kabul aldı; bilimsel sertifikasyon değildir. Diğer profillerin sonuçları validation/mirt_stage1 raporundadır.')
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
 model=st.selectbox('MIRT modeli',MODELS+factorized.MODELS,index=MODELS.index('2PL'),key='md_model')
 restricted=model in factorized.MODELS;c=.2;u=1.
 if restricted:
  st.warning(factorized.LIMITATION)
  c=st.number_input('Sabit alt asimptot c (kestirilmez)',min_value=0.,max_value=.5,value=.2,step=.01,key='md_fixed_c')
  if model.startswith('4PL'):u=st.number_input('Sabit üst asimptot d (kestirilmez)',min_value=.5,max_value=1.,value=.95,step=.01,key='md_fixed_u')
 dims=st.selectbox('Boyut sayısı',[2,3,4,5],key='md_dims')
 mode=st.selectbox('Yapı',['Doğrulayıcı','Keşfedici (pozitif, üçgen)'],key='md_mode')
 efa=mode.startswith('Keşfedici');corr=st.selectbox('Boyut korelasyonları',['Ortogonal','Yalnız F1–F2','Tümü'],key='md_corr')
 correlated={'Ortogonal':False,'Yalnız F1–F2':True,'Tümü':'full'}[corr]
 pattern=np.zeros((len(items),dims),int)
 if len(items):pattern[np.arange(len(items)),np.minimum(np.arange(len(items))*dims//len(items),dims-1)]=1
 if efa:
  pattern[:,:]=1;pattern[:min(dims,len(items)),:]=np.tril(np.ones((dims,dims),int))[:len(items),:]
 if model=='Bifactor':
  groups=st.text_input('Özgül gruplar (madde sırasıyla 0,1,2…)',','.join(str(i*max(dims-1,1)//max(len(items),1)) for i in range(len(items))),key='md_groups')
  chosen=groups
 else:
  matrix=st.data_editor(pd.DataFrame(pattern,index=items,columns=[f'F{j+1}' for j in range(dims)]),key=f'md_pattern_{len(items)}_{dims}_{efa}',num_rows='fixed')
  chosen=matrix.to_numpy().tolist()
 labels=None
 if model=='NRM':
  observed=list(dict.fromkeys(str(v).strip().upper() for v in data[items].to_numpy().ravel())) if items else []
  labels=tuple(v.strip().upper() for v in st.text_input('Nominal etiketler: kod sırası, başarı sırası değil',','.join(observed),key='md_labels').split(','))
 nodes=st.selectbox('Eksen başına GH düğümü',[21,41,61] if restricted else {2:[9,15,21],3:[5,9,15],4:[5,7],5:[5]}[dims],key=f'md_quad_{dims}')
 iterations=st.number_input('Maksimum iterasyon',min_value=1,max_value=3000,value=1000,step=100,key='md_iter')
 tolerance=st.number_input('EM toleransı',min_value=1e-10,max_value=1e-3,value=1e-7,format='%.10g',key='md_tol')
 settings=dict(c=c,u=u,items=items,person=person,model=model,dims=dims,efa=efa,correlated=correlated,pattern=chosen,labels=labels,n_quadpts=nodes,max_iter=int(iterations),tol=tolerance)
 key=fingerprint(data,settings);saved=st.session_state.get(KEY)
 if saved and saved['key']!=key:st.session_state.pop(KEY,None);saved=None
 try:
  if not items:raise ValueError('Maddeleri seçin')
  ids=data.index.tolist() if person==id_options[0] else data[person].tolist()
  if pd.Series(ids).isna().any() or not pd.Index(ids).is_unique:raise ValueError('Kimlikler eksiksiz ve benzersiz olmalı')
  if model=='Bifactor':
   chosen=[int(v.strip()) for v in chosen.split(',')]
   if len(chosen)!=len(items) or len(set(chosen))+1!=dims or efa or correlated:raise ValueError('Bifactor: ortogonal, boyut=özgül grup+1; grup başına ≥3 madde')
  elif restricted:factorized.validate(data[items],chosen,correlated,efa,c,u)
  else:structure(chosen,correlated,efa)
 except Exception as exc:st.session_state.pop(KEY,None);st.error(f'Ön kontrol: {exc}');return
 cost=f'{dims} bağımsız blok × {nodes} GH düğümü' if restricted else f'{nodes**dims} GH düğümü'
 st.caption(f'{len(data)} kişi × {len(items)} madde; {cost}. Eksik veri otomatik doldurulmaz. İşlem bütçesi aşılırsa kestirim engellenir.')
 if st.button('MIRT kestir',key='md_run',type='primary'):
  st.session_state.pop(KEY,None);saved=None
  try:
   with st.spinner('Gerçek Python mirt marjinal ML kestirimi'):
    if restricted:result=factorized.report(factorized.fit(data[items],model,chosen,correlated=correlated,exploratory=efa,c=c,u=u,n_quadpts=nodes,max_iter=int(iterations),tol=tolerance),ids)
    else:result=report(fit(data[items],model,chosen,correlated=correlated,n_quadpts=nodes,max_iter=int(iterations),tol=tolerance,labels=labels,exploratory=efa),ids)
   result['metadata']['Input_fingerprint']=key;result['metadata']['ID_column']=person
   saved=dict(key=key,result=result);st.session_state[KEY]=saved
  except Exception as exc:st.error(f'Kestirim/raporlama başarısız: {exc}')
 if not saved:return
 result=saved['result'];tables=result['tables']
 if not result['metadata']['Converged']:st.error('NOT CONVERGED — yalnız geçici parametreler; puanlar ve eğriler kapalı.')
 tabs=st.tabs(['Özet / yakınsama','Parametreler / yapı','Kişi puanları','Bilgi / SEM','Eğriler','Raporlar'])
 for tab,names in zip(tabs[:4],[['Summary','Fit'],['Parameters','Pattern','Covariance','Boundaries'],['EAP','MAP'],['SEM','IIF','TIF']]):
  with tab:
   for name in names:
    if name in tables:st.write(name);st.dataframe(tables[name],hide_index=True)
 item=items[0];category=result['metadata']['Labels'][0]
 with tabs[4]:
  if 'Curves' in tables:
   item=st.selectbox('Eğri maddesi',items,key='md_plot_item');category=st.selectbox('Kategori',list(result['metadata']['Labels']),key='md_plot_cat')
   fig=figure(result,item,category);st.pyplot(fig)
   import matplotlib.pyplot as plt
   plt.close(fig)
 with tabs[5]:
  bundle_key=(item,str(category))
  if saved.get('bundle_key')!=bundle_key:saved['bundle']=downloads(result,item,category);saved['bundle_key']=bundle_key
  for ext,mime in [('csv','text/csv'),('xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),('png','image/png'),('zip','application/zip'),('report','text/markdown')]:
   if saved['bundle'][ext] is not None:st.download_button('MIRT — '+ext.upper(),saved['bundle'][ext],'mirt_results.'+('md' if ext=='report' else ext),mime,key='md_download_'+ext)
