# DIF Analiz Merkezi — kapsam ve bağımsızlık sınırları

Taban `bc9aeff`, PR #128/#129 dahil. Yeni kestirim algoritması yok; mevcut
`core.dif_classical`, `mirt.sibtest`, public çok-grup ve eski formel DIF sayfaları
korunur. Menü: **MTK: DIF Analiz Merkezi**. Mevcut kullanıcı ekranları kaldırılmaz.

| Yöntem | Uygulama / sayısal kanıt | Bağımsız referans | Açık sınır |
|---|---|---|---|
| SIBTEST | Gerçek mirt1.2.0 original/crossing; β, SH, z, χ²/sd/p, yön; sabit çapalar | Düzeltmesiz iki-katman hesabı; düzeltilmiş yol gerçek API ile çalıştırıldı | Düzeltilmiş/crossing genel Tip-I/FWER ve etki sınıfı sertifikası yok |
| IRT-LR | Mevcut sabit-çapalı model ekranları; public mirt LR/sd/parametre farkları/eğriler | Önceki modele özgü arşivler; yerel↔Python aynı proje karşılaştırması R değildir | Public p/q/Wald kapısı kapalı; mevcut yerel model kapıları değişmedi |
| Lord χ² | Üretim public mirt çok-grup yolunda **BLOCKED** | Yeni test/benchmark yok | Ortak bağlı-ölçek parametre kovaryansı sunulmuyor; grup SH’lerini bağımsız sayıp kovaryans uydurulmaz |
| Raju | İkili, ortak c/u, aynı bağlı θ ölçeğinde tam-doğru işaretli/işaretsiz alan | Bağımsız scipy adaptif sonsuz-aralık integrali | Betimsel; SH/test/p ve sınıf yok; farklı asimptotlar ıraksar |
| MH | Mevcut MH/RBG ve ETS ΔMH/OR/A-B-C | statsmodels StratifiedTable/RBG | Koşullu gözlenen puan null’u; genel FWER sertifikası yok |
| Lojistik LR | Mevcut M0/M1/M2; uniform/nonuniform/omnibus, seçilen birincil hipotez, omnibus ΔR² | Mevcut likelihood/df kimlikleri ve gerçek statsmodels | Impact/çapa koşulları; bileşen için omnibus etki sınıfı atanmaz |
| Wald DIF | Üretim public mirt çok-grup yolunda **BLOCKED** | Genel `mirt.wald` imzası kaynak koddan doğrulandı | Tek-grup Wald, DIF değildir; ortak kovaryans, link belirsizliği ve tanı kapıları gerekir |

**Hiçbir yeni yöntem CERTIFIED değildir.** `checks.json` gerçek no-fit çalışmanın
makine-okunur sayısal sonuçları, sürümleri, API imzaları ve public çok-grup sonuç
alanlarını içerir. `python -m validation.dif_center.checks` ile yeniden üretilir;
kalibrasyon, simülasyon, RNG veya ağ çağrısı içermez. Yerel `core.irt_dif` araştırma
motorunun ortak kovaryansı olduğu bilinir; bu, public mirt kovaryansı değildir.
Yerel motoru yeniden birincil üretim kestiricisi yaparak kapılar aşılmadı.

## Analiz sözleşmesi

- Aynı yanıtlar, iki grup, sabit çapa ve ortak test ailesi; seçilen gruplar dışı ve
  eksik-grup satır sayıları kaydedilir. Eksik madde yanıtları ve ordinal/nominal
  veri bu ikili yeni yolda reddedilir. Ordinal Mantel/LR ve arındırma mevcut ayrı
  yola yönlendirilir. Çapalar otomatik doğrulanmış sayılmaz.
- MH/LR eşleştirmesi = çapa toplamı + incelenen madde; SIBTEST = yalnız çapalar.
  Null hipotezleri eşdeğer değildir; p grafiği üstünlük, eşdeğerlik veya yöntemler
  arası ortak hata kontrolü testi değildir.
- Holm/Bonferroni/BH her yöntem için seçilen hipotezin tüm planlı madde ailesine
  uygulanır. Hesaplanamayan testler NaN ve gerekçeyle saklanır; düzeltmede aile
  büyüklüğünü küçültmez. Ek LR bileşen p değerleri ham/tamamlayıcıdır.
- β_original = reference−focal. β_cross = |β_low−β_high| işaretsizdir; crossing
  p değeri z’nin normal kuyruğundan hesaplanmaz, gerçek bölgesel χ²/sd kullanılır.
- Raju A=∫(P_F−P_R)dθ=(u−c)(b_R−b_F); işaretsiz alan kesişimde iki parçadır.
  Eğimler farklıyken cebirsel logaddexp ifadesi yakın-paralel eğimlerde taşan
  kesişim hesabından kaçınır. Sonlu [-4,4] eğri grafiği tam Raju alanı değildir.
- θ duyarlılığı: tek havuzlanmış Rasch/sabit-a 1PL/2PL ML uyumu; aynı parametre,
  N(0,1) önsel ve GH61 EAP korunur. Seçilmiş yanıtlar yalnız puanlamada NaN ile
  maskelenir; yeniden kalibrasyon yok. ID, önce/sonra posterior SD ve Δθ tutulur.
  Bu, gerçek θ hatası/RMSD, seçim sonrası geçerli çıkarım veya bias düzeltmesi değildir.
- UI/CSV/XLSX/ZIP/PNG/MD aynı tabloları kullanır. Veri değerleri (aynı şekil dahil),
  grup, referans/odak, maddeler, çapalar, hipotez, düzeltme, alpha ve SIBTEST ayarları
  değişince eski tablolar/indirmeler temizlenir. θ seçimleri kendi anahtarıyla silinir.

## Gerçekleştirilen hedefli doğrulama

31 ayrı hedefli test vakası geçti. SIBTEST crossing için ilk testin β_net varsayımı hatalıydı: gerçek
mirt kaynak kodu β_cross=|β_low−β_high| döndürür. Tanım ve yön etiketi düzeltildi;
önceden tanımlı bölgesel χ²=14.5, sd=2 beklentisi ve tolerans değiştirilmedi.
İki-katman β_original=1/3, χ²=14.5, sd=1; maksimum hata 1.78e−15.
Raju adaptif integraliyle maksimum işaretli/işaretsiz hata 1.56e−15/2.18e−14; eşlenmiş tam null β=0, p=1. Bu tek
matematiksel örnek Tip-I/FWER deneyi değildir. θ posteriorları bağımsız GH61
normalizasyonuyla 1e−12 cebirsel toleransta doğrulandı; tek gerçek kalibrasyon
sayacı ve ID sırası kontrol edildi. Mevcut bağlı public 2PL LR, gerçek Streamlit
akışları, yedi stale-input vakası, tüm formel yönlendirmeler ve güvenli raporlar
çalıştırıldı. Sayısal toleranslar eşitlik/entegrasyon testidir, bilimsel kabul
sınırı veya sonradan ayarlanmış FWER ölçütü değildir.

**Rscript yok: yeni bağımsız R/difR/lordif çalışması NOT EXECUTED.** Uysal ve ark.
(2019), DOI 10.21031/epod.534312 PDF’i sağlanmadı; makale replikasyonu iddia edilmez.
GH101/GH161, 1e−5 kapısı, 4PL politikaları ve geçmiş ham kanıtlar değişmedi:
**NUMERICALLY VALIDATED — 30 PAIRED SEEDS** / **FWER NOT CERTIFIED**.
Yeni Monte Carlo veya büyük yeniden kalibrasyon yok; tek tam son regresyon CI
üzerinden istenir. Son CI durumu PR üzerinde kaydedilir. Otomatik merge yok.

Bağımlılık sırası: mevcut MH/LR → public SIBTEST → mevcut bağlı IRF’lerden Raju
→ sabit kalibrasyon EAP duyarlılığı. Sonraki en küçük bağımsız adım, aynı küçük
sabit veriyle R/difR düzeltilmiş SIBTEST karşılaştırması ve public çok-grup
ortak kovaryans/ölçek sözleşmesinin doğrulanmasıdır; sonra Lord/Wald kapısı
ayrıca değerlendirilebilir. Bu PR bunlar için yeni R kurulumu/Monte Carlo yapmaz.

## Son regresyon ve additive-output kontrolü

İlk CI (`41433659362`, `ba8d71b`) tam regresyonu: **1258 passed, 15 skipped**,
979.89 s. Tek iş hatası, eski golden-master tablosunda yeni `Raju_area` anahtarının
bulunmamasıydı. Karşılaştırıcı artık eklenen/silinen anahtarı tam çocuk yoluyla
bildirir; mevcut `--intended` mekanizması yalnız
`/python_dif/2PL/tables/Raju_area:` eklemesini tanır. Üst tablo, eski katsayılar,
LR, olasılık veya toleranslar dışlanmaz/değişmez; referans commit aynı kaldı.
Yeni test, eski LR’de tek ULP değişikliğinin ve yeni alanın kaldırılmasının hâlâ
başarısız olduğunu doğrular. Golden karşılaştırıcının **4 hedefli testi geçti**
(toplam 35 ayrı hedefli vaka). Yeni tablo mevcut bağımsız integral testleriyle
ayrıca doğrulanır. Üretim kodu tam regresyondan sonra değişmedi. İkinci tam süit
çalıştırılmaz; son başlıkta mevcut zorunlu CI kontrolleri izlenir.

Yeni koruma testi mevcut allowlist uygulamasında ek bir hata yakaladı: bulunmayan
beklenen fark için üretilen hata, aynı önek filtresiyle tekrar siliniyordu.
Filtreleme hata eklenmeden önceye taşındı; artık eski/bulunmayan izin gerçekten
FAIL üretir. Sayısal tolerans değişmedi. Dört golden testi düzeltmeden sonra geçti.
