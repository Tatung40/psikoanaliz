"""One reusable example picker, real-loader handoff and preparation guide."""
from functools import lru_cache
import streamlit as st
import pandas as pd

from core import example_datasets as E


def discard_results(state,drop_scored=True):
    """Keep inputs/widgets; discard cached analysis objects across all engines."""
    keep={'raw_df','item_cols','answer_key','id_col','group_col','_example_metadata','_example_active'}
    if not drop_scored:keep.add('scored_df')
    for key in list(state):
        if key in keep:continue
        value=state[key]
        if key=='scored_df' or not isinstance(value,(str,int,float,bool,type(None),list,tuple)):
            del state[key]
    for key,value in {'test_summary':{},'item_summary':None,'reliability_data':{},'distractor_data':{}}.items():
        state[key]=value
    if drop_scored:state['scored_df']=None


def own_file_changed(upload_key):
    """New personal file: clear the previous example, key, roles and cached results."""
    for key in list(st.session_state):
        if key not in ('nav_page','privacy_upload_nonce',upload_key):del st.session_state[key]


def key_changed():
    discard_results(st.session_state)


def activate(dataset_id,page):
    bundle=E.materialize(dataset_id)
    state=st.session_state
    nonce=int(state.get('privacy_upload_nonce',0))+1
    # Callback runs before widgets: reset old models, uploaded files and all caches.
    picker_key='examples_pick_'+page
    for key in list(state):
        if key not in ('nav_page',picker_key):del state[key]
    state.update(raw_df=E.input_frame(bundle),scored_df=None,item_cols=bundle['items'],
                 id_col=bundle['id'],group_col=bundle['group'],answer_key=bundle['key'],
                 privacy_upload_nonce=nonce,_example_active=dataset_id,_example_metadata=bundle['metadata'])
    if bundle['spec'].family!='g':
        state['scored_df']=E.scored(bundle) if bundle['key'] else bundle['data'].copy()
    observed=set(pd.Series(bundle['data'][bundle['items']].to_numpy().ravel()).dropna().unique())
    state['is_dichotomous']=bool(bundle['key']) or bool(observed) and observed <= {0,1}
    state.update(bundle['settings'])
    if bundle['spec'].model in ('GRM','PCM','GPCM','NRM'):
        state.update(b14_model=bundle['spec'].model,b14_categories='A,B,C,D',
                     grm_categories=4,poly_categories=4,poly_labels='["A","B","C","D"]')
    destination=page
    if page in ('1. Ana Sayfa','2. Veri Yükleme'):
        family=bundle['spec'].family
        destination=E.G_PAGE if family=='g' else E.M_PAGE if family=='mirt' else E.MISSING_PAGE if family=='missing' else '3. Puanlama & Veri Hazırlama' if family=='abc' else 'Boyutluluk ve Faktör Analizi' if family=='factor' else 'MTK: Python mirt '+bundle['spec'].model if bundle['spec'].model!='RSM' else 'MTK: RSM'
    if page=='6. Çeldirici Analizi':destination=page
    state['nav_page']=destination
    if destination==E.G_PAGE:state['_example_g_run']=True
    state['_example_loaded_notice']=True
    if destination==E.MISSING_PAGE and page!=E.MISSING_PAGE:
        st.switch_page('pages/90_Kayip_Veri_Analizi_ve_Atama.py')


@lru_cache(maxsize=8)
def _files(dataset_id):return E.downloads(E.materialize(dataset_id))


def guide(page,bundle=None):
    spec=bundle['spec'] if bundle else E.available(page)[0]
    if spec.family=='g':
        text=('Her satır tek birey × madde/puanlayıcı hücresidir; sütunlar seçilen desene göre p, i, r ve sayısal score. '
              'Her yüzeyde en az iki düzey, tüm hücrelerde tek gözlem ve tam denge gerekir. '
              'p × (r:i): r maddelere yuvalı; r kimlikleri I1_R1, I2_R1 gibi ebeveynler arasında benzersizdir. '
              'p × (i:r): i puanlayıcılara yuvalı; I kimlikleri ebeveynle birlikte yazılır. '
              'Eksik/dengesiz/sabit yüzeyli veri kabul edilmez. Yalnız p × i geniş dosyası güvenli dönüşüm için sunulur. '
              'D sayıları yuvalı yüzeylerde ebeveyn başınadır; birey sayısı sabittir.')
    elif spec.family=='mirt':
        text=('Her satır benzersiz katılımcı, her M sütunu madde; ogrenci_id kimliktir, grup sütunu madde değildir. '
              'Örnekteki boyut ve modeli otomatik eşlenir; madde-boyut matrisi ekranda düzenlenebilir. '
              'Boyut başına en az üç saf madde/tanımlama kısıtları gerekir; bifaktörde genel ve özgül yükler ayrılır. '
              'Örnekler ortogonal doğrulayıcı yapılardır; serbest korelasyon/within-item yapı örneği olduğu iddia edilmez. '
              'NRM etiketleri nominaldir; başarı sırası yoktur. Modelin mevcut EXPERIMENTAL / NOT CERTIFIED kısıtları aynen sürer.')
    elif spec.family=='dif' or 'DIF' in page:
        text=('Her satır katılımcı; ogrenci_id, iki düzeyli grup (Referans/Odak) ve M01… madde sütunları. '
              'Kimlik/grup sütunlarını maddelere katmayın. M01/M02 üretici tasarımda değişmez çapalardır; '
              'bu bilgi gerçek veride bağımsız kanıt gerektirir. Etki varsa yalnız M03 üzerindedir; null etki yok demektir, '
              'her örneklemde p>.05 garantisi değildir. NRM kategorileri sırasızdır. '
              'Sıralı modellerin adım/eşik ve eğim sınırları birbirinin yerine geçmez. '
              'Geçerli çıkarım/yakınsama kapıları değiştirilmez; duyarlılık ve FDR sınırlılıkları sürer.')
    elif spec.family=='abc' or page=='3. Puanlama & Veri Hazırlama':
        text=('Her satır öğrenci; ogrenci_id, grup ve M01… sütunlarında A/B/C/D/E yanıtı. '
              'İlk satır değişken adlarıdır. Ham harf yanıtları 0/1 değildir. '
              'Anahtarda madde,dogru_cevap sütunları bulunur; eşleşme madde adına göredir, sütun sırasına göre değil. '
              'Anahtarı örnekten alın, Puanlama ekranında elle düzenleyin veya ayrı CSV/XLSX yükleyin. '
              'Örnek Excel kitabında veri ilk, cevap_anahtari ayrı sayfadır. '
              'Boş yanıtı boş hücre bırakın; 0 puanlamak yalnız ilan edilen puanlama kuralıyla mümkündür. '
              'Puanlanmış dosya aynı anahtarla üretilir ve KTK/ikili MTK yoluna aktarılabilir.')
    else:
        text=('Her satır katılımcı; ogrenci_id kimlik, grup isteğe bağlı grup değişkeni, M01… analiz maddeleridir. '
              'Rasch/1PL/2PL/3PL/4PL için puanlar 0/1; GRM/PCM/GPCM/RSM için tam sayı 0–3 '
              '(1–4 de mevcut kategori eşlemesiyle mümkündür). NRM A/B/C/D gibi nominal etiket ister; '
              'sıralı eşik veya başarı puanı gibi yorumlanmaz. Sürekli faktör verisi kesirli sayısaldır. '
              'Kimlik, grup ve toplam puanı madde listesine eklemeyin. Boş hücre eksik yanıttır; '
              'sabit madde ve kategori frekanslarını kontrol edin. Eksikleri motorun kabulüne göre yönetin; '
              'çok kategorili kalibrasyonda eksikler otomatik uydurulmaz.')
    with st.expander('Kendi Veri Setimi Nasıl Hazırlamalıyım?'):
        st.write(text)
        st.write('CSV/XLSX doğrudan; TXT/TSV sekmeyle ayrılmış metindir (Veri Yükleme: otomatik ayırıcı veya sekme). '
                 'Ondalık için nokta, benzersiz başlıklar kullanın. .sav/.xls desteği isteğe bağlı paket gerektirir; bu örnekler o biçimlerde üretilmez.')
        st.text('GENİŞ İKİLİ VERİ ÖRNEĞİ\nDOĞRU: ogrenci_id,M01,M02,grup\nO001,1,0,Referans\nYANLIŞ: kimlik ve toplam puanı madde olarak seçmek')
        if spec.family=='missing':st.caption('Örnek kayıp mekanizması üretimde MCAR olarak tanımlıdır; gerçek veride MCAR/MAR/MNAR yalnız kayıp oranından belirlenemez.')


def _select_example(page):
    selected=st.session_state.get('examples_pick_'+page)
    if selected:
        activate(selected,page)


def render(page):
    choices=E.available(page)
    if not choices:return
    bundle=None
    with st.expander('Örnek Veri Setleri ve Veri Hazırlama Rehberi'):
        selected=st.selectbox('Örnek veri seti',[None]+[d.id for d in choices],
            format_func=lambda v:'Örnek seçin (veri henüz üretilmez)' if v is None else E.registry()[v].title,
            key='examples_pick_'+page,on_change=_select_example,args=(page,))
        if selected:
            bundle=E.materialize(selected)
            st.caption('SENTETİK öğretim verisi — gerçek araştırma verisi veya bilimsel sertifika değildir.')
            st.dataframe(bundle['data'].head(6),hide_index=True)
            st.write({'Kimlik':bundle['id'],'Grup':bundle['group'],'Maddeler':bundle['items'],
                      'Desen/model':bundle['spec'].design or bundle['spec'].model,'Seed':bundle['spec'].seed})
            with st.expander('Üretici parametreler ve veri sözlüğü'):st.json(bundle['metadata'])
            ext=st.selectbox('İndirme biçimi',['csv','xlsx','tsv','txt'],key='examples_format_'+page)
            files=_files(selected)
            st.download_button('Örnek veriyi indir',files['veri.'+ext],selected+'.'+ext,
                               key='examples_data_'+page,on_click='ignore')
            if bundle['key']:
                st.download_button('Ayrı cevap anahtarını indir',files['cevap_anahtari.'+ext],'cevap_anahtari.'+ext,
                                   key='examples_key_'+page,on_click='ignore')
                st.download_button('Puanlanmış 0/1 verisini indir',files['puanlanmis.'+ext],'puanlanmis.'+ext,
                                   key='examples_scored_'+page,on_click='ignore')
            st.download_button('Tüm sayfalar ve cevap anahtarı — XLSX',files['tum_sayfalar.xlsx'],selected+'.xlsx',
                               key='examples_book_'+page,on_click='ignore')
            if 'genis.csv' in files:st.download_button('Geniş p × i — CSV',files['genis.csv'],selected+'_genis.csv',key='examples_wide_'+page,on_click='ignore')
            st.download_button('Üretim bilgileri — JSON',files['uretim_bilgileri.json'],selected+'.json',key='examples_meta_'+page,on_click='ignore')
            st.caption('Örnek seçildiğinde veri otomatik yüklenir ve alanlar eşlenir. G/D ve otomatik KTK ekranları hesaplar; '
                       'diğer modellerde ayarları kontrol edip ilgili analizi başlatın. Yakınsama garanti edilmez.')
    guide(page,bundle)
