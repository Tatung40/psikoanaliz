# Örnek veri kütüphanesi v1.0

Tek registry: `core/example_datasets.py`; tek bileşen:
`ui/example_datasets_view.py`. Güncel main `0e8bb17` üzerindeki **40 analiz
ekranı** (bağımsız eksik veri sayfası dahil), MIRT ekranındaki **15 mevcut model
profili** ve DIF/puanlama alt seçenekleri için kaynak API'lerine göre rehber
ve filtrelenmiş örnekler sağlanır. Envanter testi gerçek `selected_page`
yollarını denetler; uygulamada ayrı DFA/regresyon ekranı varmış gibi gösterilmez.

## Kapsam: 117 örnek

| Aile | Sayı | Anlamlı senaryolar |
|---|---:|---|
| Tek boyutlu MTK | 30 | Rasch, 1PL, 2PL, 3PL, 4PL, GRM, PCM, GPCM, NRM, RSM × üç düzey |
| Mevcut MIRT profilleri | 45 | 15 profil × küçük 2D, daha büyük 3D (izin verilenlerde), uzun test; bifaktör 3D |
| DIF | 18 | 2PL/GRM/PCM/GPCM/RSM/NRM × null, tek biçimli, eğim veya daha güçlü konum etkisi |
| G-Kuramı | 15 | Etkin beş desen × temel/gerçekçi/ileri |
| ABCDE | 3 | 4/20/50 madde; ham veri + aynı anahtardan türeyen 0/1 veri + ayrı anahtar |
| Sürekli faktör verisi | 3 | İki faktör, farklı N/test uzunluğu |
| Eksik veri | 3 | Bağımsız Bernoulli MCAR %3/%10/%20; bunlar gerçek mekanizma tanısı değildir |

Uzantılar ayrı veri seti sayılmaz. Geniş p×i, anahtar ve puanlanmış tablolar
aynı örneğin eşlenmiş türevleridir; toplam sayıya tekrar eklenmez.

## Bilimsel anlam ve doğrulama

Veri üretimi **Python mirt 1.2.0 `simdata`** kamusal API'sini kullanır;
RSM için mevcut `RatingScaleModel.set_parameters` ile ortak adımlar tanımlanır.
Yeni kestirim algoritması veya R kod üretimi yoktur. `simdata('1PL')` eğimi
1'e sabitlediğinden serbest ortak a=1.2 öğretim örneği 2PL üretici üzerinde
eşit eğimlerle oluşturulur. PCM/GPCM `steps` kullanır; difficulty ile adım
parametreleri birbiri yerine geçirilmez. NRM kategori eğimleri sıralı değildir.

Her örnekte seed, N/k, üretici parametreler, kategori anlamı, NumPy/mirt sürümü
ve gerektiğinde boyut yükleri/korelasyonları kaydedilir. Veriler ihtiyaç halinde
üretilir; açılışta bütün matrisler veya kalibrasyonlar oluşturulmaz.
Çekilişlerde yetenek RNG'si yeniden başlatılmaz: yanıt akışı
`SeedSequence([call_seed,104729])` ile ayrılır; grup/madde seed ofsetleri de
kaynakta ve üretim bilgisinde açıkça belirtilir.
İndirme önbelleği yalnız sekiz sentetik örneğe ait byte dizilerini tutar; kişisel veri
global önbelleğe alınmaz. Her `materialize` çağrısı yeni ve düzenlenebilir
frame döndürür. Büyük statik veri eklenmez.

**117 örneğin hepsi gerçek CSV yükleyiciyle şema/tekrarlanabilirlik testinden
geçmek zorundadır; bu, 117 bağımsız kalibrasyonun geçtiği anlamına gelmez.**
Hedefli testler temel 2PL/GRM/PCM/GPCM/NRM örneklerinde gerçek Python mirt
kalibrasyonunu ve yakınsamayı ayrıca sınar. 3PL/4PL ve MIRT örnekleri otomatik
kalibre edilmez; mevcut ekranlarda kullanıcı ayarları onaylayıp çalıştırır.
Yakınsama/parametre kurtarma/Tip-I hata sertifikası veya yeni R karşılaştırması
iddia edilmez. Mevcut bağımsız R mirt testleri, CI ve çıkarım kapıları korunur.

### G-Kuramı kabulü

Beş gerçek etkin desen: p×i, p×r, p×i×r, p×(r:i), p×(i:r).
Her birinde p=6/12/20; i=3/4/5 ve r=2/3/4 (desende varsa).
İleri örnekte madde SD'si de değişir. Birey SD=3, hata SD=.6; uzun
dengeli çapraz/yuvalanmış tablolar üretilir. Yuvalı kimlikler ebeveynler arasında
benzersizdir. D senaryoları her yüzey için mevcut ve +1 düzeyi içerir; p sabittir.
15 örneğin tamamı **gerçek GeneralizIT 0.1.2 G ve D API'leriyle** hedefli testte
hesaplanır. Bu **LIBRARY TESTED** durumudur, yeni bağımsız sayısal sertifika
değildir. Sabit/dengesiz/eksik desenler eklenmez. p×i geniş türevi sunulur;
diğer geniş biçimler destekleniyormuş gibi sunulmaz.

### DIF

İki grubun üretici θ dağılımı aynı N(0,1)'dir. Null koşulda hiçbir parametre
değişmez; etki varsa yalnız M03 değişir, M01/M02 üretimde değişmez çapalardır.
Sıralı b/adım konum artışı .6; güçlü PCM/RSM konum etkisi 1; eğim çarpanı 1.5;
NRM ikinci kategori kesişim/eğim artışı .6. PCM/RSM'ye sahte serbest eğim
eklenmez. Bu bilinen üretici çapa, gerçek veride bağımsız çapa kanıtı yerine
geçmez. Bootstrap ekranındaki örnekler açıkça 2PL üreticilidir; bunlar 3PL/4PL
p değerlerine yeni uygunluk sağlamaz. Null örnekte p>.05 garantisi yoktur.

## Dosya ve kullanıcı akışı

Ortak bileşen bütün analizlerden önce görünür. Örnek seç → ilk satırlar/sözlük →
CSV/XLSX/TXT/TSV indir → **Bu Örnekle Analiz Yap** gerçek `load_dataset`
üzerinden yükler ve alan/model/desenleri eşler. G/D otomatik hesaplanır;
otomatik KTK sayfaları mevcut şekilde çalışır. Diğer kestirimler mevcut açık
çalıştır düğmesine bırakılır; örnek seçimi sonsuz fit/retry başlatmaz.
Tüm örneklerde parametreler düzenlenebilir ve bilimsel kapılar geçerlidir.

Anahtarlar madde adına göre doğrulanır; eksik/fazla/yinelenen ad, yinelenen
başlık, A–E dışı seçenek veya sayısal ham veri güvenli biçimde reddedilir.
Elle giriş, ayrı CSV/XLSX ve örnekten otomatik anahtar desteklenir. Excel
kitabında `veri`, `cevap_anahtari`, `puanlanmis` sayfaları bulunur; anahtar
okuyucu isimli sayfayı seçebilir. İlk üç 4-madde satırı talep edilen A/C/E/B
örneğiyle aynıdır. 20/50 maddelik ham örnekler gerçek KTK özet/güvenirlik
API'lerinden geçirilir. Grup/kimlik sütunları puanlamadan sonra korunur.

Örnek yükleme callback'i eski dosya/widget/model/sonuç durumunu temizler;
yeni kişisel dosya da eski örnek ve anahtarı kaldırır. Anahtar değişiminde
puanlanmış veri ve analiz nesneleri temizlenir. Mevcut veri/model parmak izi
kapıları korunur; eksik veri laboratuvarına da veri/yöntem değişiminde sonuç
iptali eklenir. İndirmeler `on_click='ignore'`; URL/oturum gezinme düzeltmesi
korunur. TXT/TSV ana veri dosyaları CSV yükleyicisinin sekme ayracıyla okunur;
.sav/.xls örneği veya kurulu olmayan isteğe bağlı paket desteği uydurulmaz.

```bash
python -m pytest -q tests/test_example_datasets.py tests/test_example_datasets_ui.py
```

Test kanıtı PR/CI loglarındadır. Yeni örneklerin R karşılaştırması yapılmadı;
MIRT serbest korelasyon/within-item örnekleri, otomatik zorlayıcı kalibrasyonlar
ve bağımsız model kurtarma çalışmaları v1.0 kapsamında tamamlanmış sayılmaz.
