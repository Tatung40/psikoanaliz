# G-Kuramı: sıkçı genişleme

Önkayıt: `29dd0c2bede3e9cd29114537321bffdb9843aed5`. GeneralizIT 0.1.2 ve PR #151'in ölçek düzeltmesi değiştirilmedi.
Bu modül genel bir **CERTIFIED** iddiası vermez. NUMERICALLY VALIDATED yalnız raporda belirtilen fikstür×çıktı hücresidir.

## Kullanım ve veri

G-Kuramı ekranında mevcut uzun/geniş tek değişkenli akış korunur. Her satır tek puanlama olayıdır:
`Person | Item | Rater | Dimension | Score`; yalnız ilgili sütunlar gerekir. Roller ekranda sütunlara eşlenir.
`Dimension` nominal alt ölçek etiketidir, kişi veya madde değildir. `Score` sonlu sayısal gözlemdir.
Yuvalı alt kimlikler ebeveynler arasında benzersiz olmalıdır. Yinelenen hücreler ve eksik puanlar reddedilir;
eksik gözlem hücreleri yalnız tanımlanabilir REML modelinde kabul edilir. Sessiz satır silme yapılmaz.

`examples/manifest.json` deterministik CSV'lerin SHA-256 ve desenlerini kaydeder. Ekrandaki örnekler CSV/XLSX indirilebilir.
Çok değişkenli örnek için veri biçimi **Çok değişkenli (uzun)** seçilir; dört rol atanır, analiz etkinleştirilip çalıştırılır.

## Motor seçimi

- **auto:** geçerli dengeli beş desende mevcut ANOVA; uygun dengesiz/eksik hücrelerde REML.
- **ANOVA:** GeneralizIT; eski negatif-bileşen politikası ve 25 senaryo sınırı korunur.
- **REML:** ayrı R `lme4::lmer` işlemi; `lme4` DESCRIPTION sürümü **2.1-0** (`packageVersion` gösterimi **2.1.0**),
  Gaussian rastgele etkiler, REML, sabit kesişim; bobyqa/maxfun=100000. R kurulu değilse açık hata; sessiz ANOVA ikamesi yok.

İsteğe bağlı R bağımlılıkları kullanıcı/yönetici tarafından kurulmalıdır; uygulama paket indirmez:

```r
install.packages(c("lme4", "jsonlite"))
stopifnot(packageVersion("lme4") == package_version("2.1-0"))
```

R sürümü, Matrix/lme4/jsonlite sürümleri, gerçek veri hash'i, rol haritası, merkez/ölçek ve R betik hash'i sonuçta saklanır.
REML logLik puan ölçeğine **N−rank(X)=N−1** artık kontrastıyla dönüştürülür; N ile dönüştürülmez.
REML AIC/BIC genel ML/farklı sabit-etki modeli karşılaştırması diye yorumlanamaz.

## Bilimsel sözleşme

Beş desen: p×i, p×r, p×i×r, p×(r:i), p×(i:r). Birey ölçme nesnesi; tüm yüzeyler rastgeledir.
Tek gözlem/hücrede en yüksek etkileşim artıkla karışıktır; ayrıca bağımsız artık bileşeni uydurulmaz.
Varyans çekirdeklerinin kesişime dik projeksiyondaki Gram matrisi rank/koşul kontrolünden geçmelidir.
Yakınsamama, tekillik veya geçersiz Hessian'da ham varyans/tanı korunur; G/Phi/D çıkarımı kapatılır.
Eksiklik seçimi yok sayılabilir olmalıdır; MNAR/seçici eksiklik ve rastgele-etki dağılımı yanlışlığı için kanıt yoktur.

**Dengesiz REML sonuçlarındaki katsayılar yeni, dengeli bir D hedefi içindir.** Gözlenen eşit olmayan birey
ortalamalarının güvenirliği değildir. Yuvalı D sayısı ebeveyn başınadır; başlangıç hedefi en küçük gözlenen alt düzeydir
ve kullanıcı bunu değiştirebilir. Fazladan veri üretilmez veya eksik gözlemler doldurulmaz.

Profil GA, R'nin varyans parametreleri için profil-olabilirlik SD uçlarının karesidir; yalnız başarılı uyum/profilde verilir.
Birey rastgele etkileri ve koşullu SD'leri BLUP tanısıdır; IRT theta, parametre SE'si veya katsayı GA değildir.

## Çok değişkenli ve bileşik analiz

Yalnız dengeli/tam birey×tek rastgele yüzey, 2–5 eşlenmiş boyut. Aynı olaylar her boyutta gerekir.
Çapraz çarpım ortalama karelerinden **kovaryans** bileşenleri kestirilir; bağımsız tek değişkenli REML uyumları
birleştirilerek tam çok değişkenli model gibi sunulmaz. Ham matrisler korunur. PSD olmayan bileşende katsayı yoktur;
özdeğer kırpma veya PSD projeksiyonu yapılmaz. R bağımsız karşılaştırması iki boyutlu kayıtlı örnekle sınırlıdır.

Eşit ağırlık varsayılan, ağırlık toplamı 1; negatif ağırlık açık izin ister. Bileşik varyanslar `w'Vw` ile hesaplanır.
Sınırsız ağırlıklar pozitif tanımlı toplam kovaryansta genelleştirilmiş özdeğer çözümüdür. Toplamı sıfır özvektör
toplamı 1 ağırlıkla temsil edilemez ve reddedilir. Negatif olmayan SLSQP çözümü **EXPERIMENTAL / LOCAL OPTIMUM ONLY**;
global optimum veya bağımsız genel geçerlilik iddiası yoktur. Tekil/iyi koşullu olmayan toplam matrisle optimizasyon yoktur.

Phi-cut açık bir kare kayıp modelidir:
`[U + (mu − cut)^2] / [U + (mu − cut)^2 + absolute_error]`.
Standart Phi, kesimin üstünde olma olasılığı veya sınıflama doğruluğu değildir. Bilinmeyen parametre için GA üretmez.

## Tasarım araması ve raporlar

Ayrı analitik D araması, model yeniden kestirilmeden, 2–100 yüzey düzeyinde ve ≤10000 adayla yapılır.
Maliyet: `olay_maliyeti × birey_başına_olay + Σ yüzey_sayısı × yüzey_kurulum_maliyeti`.
API yüzey kurulum maliyetlerini, UI olay maliyeti/bütçeyi destekler. Birey sayısı sabittir; öneri yalnız aranan sınırlar içindedir.
Pareto tablosu maliyeti azaltıp G/Phi'yi artırma hedeflerini açıkça gösterir; hedefe ulaşılamaması gizlenmez.

Ortak Word/Python yakalama ve CSV/XLSX/PNG/ZIP korunur. Bileşik/arama/ağırlık çıktıları kendi ayar ve veri parmak izine bağlıdır;
girdi değişince eski karar/indirme bastırılır. R tanıları, profil uçları ve rastgele etkiler mevcut G/D ZIP/XLSX'e eklenir.

## Doğrulama ve yeniden üretme

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_gtheory_advanced.py tests/test_gtheory_advanced_ui.py
REQUIRE_GT_ADV_R=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_gtheory_advanced.py::test_independent_r_reml_mapping_and_balanced_mom tests/test_gtheory_advanced.py::test_independent_r_multivariate_polarization tests/test_gtheory_advanced.py::test_r_profile_and_affine_and_singular_diagnostics --basetemp=gtheory-advanced-diagnostics
python -m validation.gtheory_advanced.archive --collect gtheory-advanced-diagnostics --output gtheory-advanced-evidence.json
# Depoya kanıt eklendiğinde, kestirim yapmadan:
python -m validation.gtheory_advanced.archive --audit validation/gtheory_advanced/evidence.json
```

R referansı aynı standartlaştırılmış gözlemlerde doğrudan lmer API'sini kullanır; ANOVA ile yalnız dengeli iç çözümler
karşılaştırılır. Çok değişkenli bağımsız R yolu, tek değişkenli aov bileşenlerini toplam/fark polarizasyonuyla birleştirir;
Python matris kodunu kopyalamaz. Eski aov/gtheoryr arşivleri değiştirilmez.

## Açılmayan yöntemler ve lisans

Sabit yüzeyler, genel karma/yuvalı çok değişkenli REML, keyfi evren/hata kümeleri ve desteklenmeyen toplama biçimleri kapalıdır.
PRMSE/VAR tanımlarının gözlenen bileşik ile gerçek bileşik kestirimi ayrımı, yayımlanmış bağımsız yöntemle eşlenmeden açılmaz.
facet dokümantasyonuna tek başına dayanarak yöntem doğrulanmış sayılmaz. Kaynaklar: Brennan (2001), Shavelson & Webb (1991),
Haberman (2008). Ayrıntılı lisans/bağımlılık kararı: `GAP_AND_LICENSE.md`.

facet 0.0.11 @0542448, zorunlu dışlanan bağımlılıklar nedeniyle **NOT EXECUTED**. GPL kaynak uygulamaya kopyalanmadı.
Hiçbir Bayesian algoritma/bağımlılık veya Monte Carlo/bootstrap eklenmedi. Canlı sunucuda R kurulu olduğu varsayılmaz.
