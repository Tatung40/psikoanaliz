"""Advanced G-theory panels; cached outputs/exports invalidate on actual inputs."""
import io
import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core import generalizability_advanced as ga
from core.python_mirt_dif import fingerprint
from ui.dimensionality_localfit_view import _safe, _xlsx, _zip, _png_lines


def _downloads(tables, settings, prefix):
    files = {k+'.csv':_safe(v).to_csv(index=True).encode('utf-8-sig') for k,v in tables.items()}
    files['tables.xlsx'] = _xlsx(tables)
    files['settings.json'] = json.dumps(settings,ensure_ascii=False,indent=2,default=str).encode()
    st.download_button('XLSX indir',files['tables.xlsx'],prefix+'.xlsx',key=prefix+'_xlsx',on_click='ignore')
    st.download_button('ZIP (CSV/XLSX/ayarlar)',_zip(files),prefix+'.zip',key=prefix+'_zip',on_click='ignore')


def render(data, result=None):
    st.subheader('Veri setimi nasıl hazırlamalıyım?')
    st.info('Her satır bir puanlama olayıdır. Birey = ölçme nesnesi; madde/görev/puanlayıcı = yüzey; '
            'Dimension = alt ölçek; Score = gözlenen sayısal puan. Çok değişkenli yolda aynı birey×yüzey olayları '
            'her boyutta bulunmalıdır. Yuvalı alt yüzey kimlikleri ebeveynler arasında benzersiz olmalıdır.')
    with st.expander('Deterministik örnekler ve bilimsel kapsam',expanded=False):
        design = st.selectbox('Örnek desen',list(ga.gt.DESIGNS),key='gt_example_design')
        ex = ga.example('balanced')
        if design=='p x i':ex=ex[ex.r==0].drop(columns='r')
        if design=='p x r':ex=ex[ex.i==0].drop(columns='i')
        if design=='p x (r:i)':ex=ex.assign(r=ex.i*3+ex.r)
        if design=='p x (i:r)':ex=ex.assign(i=ex.r*4+ex.i)
        ex=ex.iloc[1:].reset_index(drop=True)
        ex=ex.rename(columns={'p':'Person','i':'Item','r':'Rater','score':'Score'})
        st.download_button('Eksik hücreli REML örneği CSV',ex.to_csv(index=False).encode(),
                           'G_REML_'+design.replace(' ','')+'.csv',key='gt_example_reml_csv',on_click='ignore')
        st.download_button('Eksik hücreli REML örneği XLSX',_xlsx({'Veri':ex}),
                           'G_REML_ornek.xlsx',key='gt_example_reml_xlsx',on_click='ignore')
        mv = ga.example('multivariate').rename(columns={'p':'Person','i':'Item','score':'Score'})
        st.download_button('Çok değişkenli örnek CSV',mv.to_csv(index=False).encode(),
                           'G_multivariate.csv',key='gt_example_mv_csv',on_click='ignore')
        st.download_button('Çok değişkenli örnek XLSX',_xlsx({'Veri':mv}),
                           'G_multivariate.xlsx',key='gt_example_mv_xlsx',on_click='ignore')
        st.caption('Yerel R yoksa REML NOT EXECUTED. facet doğrudan kurulmaz: zorunlu dışlanan bağımlılıkları vardır. '
                   'Sabit yüzeyler, genel çok değişkenli REML ve keyfi evren/hata tanımları UNSUPPORTED. '
                   'PRMSE/VAR: bağımsız yöntem tanımı doğrulanmadı; hesap ve sertifikasyon kapalı.')
    if result is not None:
        if 'diagnostics' in result:
            st.subheader('REML tanıları')
            st.dataframe(result['diagnostics'])
            st.warning('Tekil veya yakınsamayan çözümden katsayı çıkarılmaz. BLUP birey puanı IRT theta değildir; '
                       'koşullu SD, varyans bileşeninin SE/GA ölçüsü değildir.')
            st.dataframe(result['random_effects'])
            people=result['random_effects'].query("term == 'p'")
            st.plotly_chart(px.scatter(people,x='deviation',y='original_label',error_x='conditional_sd',
                title='Birey rastgele etkileri ±1 koşullu SD (parametre GA değil)'),use_container_width=True)
            if len(result['profile_ci']):
                st.caption('Varyans bileşenlerinin R profil-olabilirlik GA uçları; SD uçları karesine dönüştürülmüştür.')
                st.dataframe(result['profile_ci'])
        if result['status'] in ('LIBRARY TESTED','NUMERICALLY VALIDATED'):
            st.subheader('D tasarım optimizasyonu / duyarlılık')
            upper={f:int(st.number_input(f'Arama üst sınırı {f}',2,100,max(10,n),key='gt_search_'+f))
                   for f,n in result['counts'].items() if f!='p'}
            target=float(st.number_input('Arama hedefi (G ve Phi)',.01,1.,.8,.01,key='gt_search_target'))
            cost=float(st.number_input('Olay başına maliyet',0.,100000.,1.,key='gt_search_cost'))
            maxcost=float(st.number_input('Birey başına bütçe (0 = sınır yok)',0.,1000000.,0.,key='gt_search_budget'))
            searchkey=fingerprint(data,(st.session_state.get('gt_key'),upper,target,cost,maxcost))
            if st.session_state.get('gt_search_key')!=searchkey:
                st.session_state.pop('gt_search_result',None);st.session_state['gt_search_key']=searchkey
            if st.button('Sınırlı D aramasını çalıştır',key='gt_search_run'):
                st.session_state.pop('gt_search_result',None)
                try:
                    ranges={'p':[result['counts']['p']],**{f:list(range(2,n+1)) for f,n in upper.items()}}
                    st.session_state['gt_search_result']=ga.optimize_design(result,ranges,target,target,cost,budget=maxcost or None)
                except ValueError as exc:st.error(str(exc))
            search=st.session_state.get('gt_search_result')
            if search:
                st.caption('Yalnız girilen sınırlar içindeki en ucuz dengeli hedef. Yeni kalibrasyon yapılmaz; evrensel tasarım sertifikası değildir.')
                st.dataframe(search['recommendation'])
                if search['recommendation'].empty:st.info('Bu sınırlar/bütçe/hedefler içinde uygun tasarım yok.')
                st.dataframe(search['candidates'])
                st.plotly_chart(px.scatter(search['candidates'],x='cost_per_person',y='phi^2',color='pareto',
                                           title='Maliyet / güvenirlik ve Pareto sınırı'),use_container_width=True)
                _downloads({k:search[k] for k in ('candidates','recommendation')},search['settings'],'gt_optimization')
            if st.checkbox('Kare kayıp modelinde Phi-cut göster',value=False,key='gt_cut_show'):
                cut=float(st.number_input('Kesim puanı',value=0.,key='gt_cut'))
                mean=result['settings'].get('model_mean',result['settings'].get('library_input',{}).get('center'))
                row=result['errors'].iloc[0]
                st.metric('Phi-cut (kare kayıp)',f"{ga.phi_cut(row.universe,row.absolute_error,mean,cut):.6f}")
                st.caption('Standart Phi veya sınıflama doğruluğu değildir; beklenen kareli kesim uzaklığı oranıdır. Parametre GA üretmez.')
    st.subheader('Çok değişkenli G — dengeli birey × tek rastgele yüzey')
    enabled=st.checkbox('Çok değişkenli analizi etkinleştir',value=False,key='gt_mv_enable')
    if not enabled:
        st.session_state.pop('gt_mv_result',None);return
    cols=list(data.columns)
    selected={}
    for label,key,defaults in [('Birey','person',('p','Person')),('Yüzey','facet',('i','Item')),
                               ('Boyut','dimension',('Dimension','dimension')),('Puan','score',('score','Score'))]:
        default=next((c for c in defaults if c in cols),cols[0])
        selected[key]=st.selectbox('Çok değişkenli '+label,cols,index=cols.index(default),key='gt_mv_'+key)
    n=int(st.number_input('Bileşik D yüzey sayısı',2,100,4,key='gt_mv_n'))
    text=st.text_input('Boyut ağırlıkları (boş = eşit; sütun sırası sonuçta gösterilir)',key='gt_mv_weights')
    negative=st.checkbox('Negatif ağırlıklara açıkça izin ver',value=False,key='gt_mv_negative')
    key=fingerprint(data,(selected,n,text,negative))
    if st.session_state.get('gt_mv_key')!=key:
        st.session_state.pop('gt_mv_result',None);st.session_state['gt_mv_key']=key
    if st.button('Çok değişkenli G çalışmasını çalıştır',key='gt_mv_run'):
        st.session_state.pop('gt_mv_result',None)
        try:
            raw=ga.multivariate(data,**selected)
            weights=[float(x.strip()) for x in text.split(',')] if text.strip() else None
            coefficients=ga.composite(raw,n,weights,negative)
            st.session_state['gt_mv_result']={'raw':raw,'coefficients':coefficients}
        except (ValueError,TypeError,KeyError) as exc:st.error(str(exc))
    result_mv=st.session_state.get('gt_mv_result')
    if result_mv:
        raw=result_mv['raw'];c=result_mv['coefficients']
        st.caption('Matris MoM; ayrı tek değişkenli REML uyumları değildir. Boyut sırası: '+', '.join(raw['dimensions']))
        st.caption('Evren ve hata kovaryans bileşenleri aynı birey/yüzey olaylarından birlikte kestirilmiştir. '
                   'Katsayı güven aralığı ve genel sertifika verilmez; bağımsız sayısal kanıt yalnız kayıtlı fikstür kapsamındadır.')
        tables={k:v for k,v in c.items() if isinstance(v,pd.DataFrame)}
        tables.update({name+'_raw':pd.DataFrame(v,index=raw['dimensions'],columns=raw['dimensions']) for name,v in raw['matrices'].items()})
        titles={'dimension_coefficients':'Boyut katsayıları','composite':'Bileşik katsayılar',
                'universe_covariance':'Evren puanı kovaryansı','relative_error_covariance':'Göreli hata kovaryansı',
                'absolute_error_covariance':'Mutlak hata kovaryansı'}
        for name,table in tables.items():
            st.subheader(titles.get(name,name));st.dataframe(table)
        st.plotly_chart(px.imshow(c['universe_covariance'],text_auto='.3f',title='Evren puanı kovaryansı'),use_container_width=True)
        files={k+'.csv':_safe(v).to_csv(index=True).encode('utf-8-sig') for k,v in tables.items()}
        files['covariance.png']=_png_lines(list(range(len(raw['dimensions']))),
            {name:np.diag(m) for name,m in raw['matrices'].items()},'Boyuta göre ham varyans','Boyut','Varyans')
        files['tables.xlsx']=_xlsx(tables)
        settings={**raw['settings'],'weights':c['weights'].tolist(),'n_facet':n,'status':raw['status'],
                  'limitations':'Balanced complete p x one random facet only; no coefficient CI or blanket certification'}
        files['settings.json']=json.dumps(settings,ensure_ascii=False,indent=2).encode()
        st.download_button('Çok değişkenli XLSX indir',files['tables.xlsx'],'G_multivariate.xlsx',key='gt_mv_xlsx',on_click='ignore')
        st.download_button('Çok değişkenli ZIP (CSV/XLSX/PNG)',_zip(files),'G_multivariate.zip',key='gt_mv_zip',on_click='ignore')
        constrained=st.checkbox('Ağırlık optimizasyonunda negatif ağırlıkları engelle',value=True,key='gt_mv_opt_nonnegative')
        absolute=st.checkbox('Mutlak Phi ağırlık hedefi',value=True,key='gt_mv_opt_absolute')
        optkey=fingerprint(data,(key,constrained,absolute))
        if st.session_state.get('gt_mv_opt_key')!=optkey:
            st.session_state.pop('gt_mv_opt_result',None);st.session_state['gt_mv_opt_key']=optkey
        if st.button('Ağırlıkları optimize et',key='gt_mv_opt_run'):
            st.session_state.pop('gt_mv_opt_result',None)
            try:
                st.session_state['gt_mv_opt_result']=ga.optimize_weights(raw,n,absolute,constrained)
            except ValueError as exc:st.error(str(exc))
        opt=st.session_state.get('gt_mv_opt_result')
        if opt:
            table=pd.DataFrame({'dimension':raw['dimensions'],'weight':opt['weights']})
            st.dataframe(table)
            st.caption(opt['status']+f" · Hedef katsayı={opt['coefficient']:.6f}")
            if constrained:st.warning('Sınırlı SLSQP yalnız yerel optimumdur; genel optimalite/sertifika iddiası yoktur.')
            _downloads({'weights':table},{**settings,**{k:v for k,v in opt.items() if k!='weights'}},'gt_mv_optimization')
