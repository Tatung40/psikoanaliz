# DIF Analiz Merkezi — kapsam ve bağımsızlık sınırları

Taban `bc9aeff`, PR #128/#129 dahil. Yeni kestirim algoritması yok; mevcut
`core.dif_classical`, `mirt.sibtest`, public çok-grup ve eski formel DIF sayfaları
korunur. Menü: **MTK: DIF Analiz Merkezi**. Mevcut kullanıcı ekranları kaldırılmaz.

| Yöntem | Uygulama / sayısal kanıt | Bağımsız referans | Açık sınır |
|---|---|---|---|
| SIBTEST | Gerçek mirt1.2.0 original/crossing; β, SH, z, χ²/sd/p, yön; sabit çapalar | Düzeltmesiz iki-katman hesabı; düzeltilmiş yol gerçek API ile çalıştırıldı | Düzeltilmiş/crossing genel Tip-I/FWER ve etki sınıfı sertifikası yok |
| IRT-LR | Mevcut sabit-çapalı model ekranları; public mirt LR/sd/parametre farkları/eğriler | Önceki modele özgü arşivler; yerel↔Python aynı proje karşılaştırması R değildir | Public p/q/Wald kapısı kapalı; mevcut yerel model kapıları değişmedi |
| Lord χ² | **2PL: fixture-specific numerical validation; 1PL: free-SD Rasch reference gate** | Gerçek R mirt1.48 sayısal uyumu; tam anchor-Jacobian | Eski sabit-σ 1PL: p/q kapalı. Yeni serbest-σ Rasch: yalnız gerçek R kanıtı + sayısal kapılarla koşullu çıkarım; genel FWER sertifikası yok |
| Raju | İkili, ortak c/u, aynı bağlı θ ölçeğinde tam-doğru işaretli/işaretsiz alan | Bağımsız scipy adaptif sonsuz-aralık integrali | Betimsel; SH/test/p ve sınıf yok; farklı asimptotlar ıraksar |
| MH | Mevcut MH/RBG ve ETS ΔMH/OR/A-B-C | statsmodels StratifiedTable/RBG | Koşullu gözlenen puan null’u; genel FWER sertifikası yok |
| Lojistik LR | Mevcut M0/M1/M2; uniform/nonuniform/omnibus, seçilen birincil hipotez, omnibus ΔR² | Mevcut likelihood/df kimlikleri ve gerçek statsmodels | Impact/çapa koşulları; bileşen için omnibus etki sınıfı atanmaz |
| Wald DIF | **IMPLEMENTED / fixture-specific VALIDATED**, 1PL M1 / 2PL M2 | Gerçek R mirt1.48, üç B10 fikstürü, mevcut ortak MML kovaryansı | 3PL/4PL yok; tek-grup `mirt.wald` ile karıştırılmaz; genel FWER sertifikası yok |

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

**İlk SIBTEST/Raju aşamasında yerel Rscript yoktu; o yöntemlerin R/difR/lordif karşılaştırması NOT EXECUTED.** Lord/Wald için aşağıdaki gerçek CI-R kanıtı ayrıdır. Uysal ve ark.
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

## Lord χ² ve Wald DIF — 1PL/2PL tamamlaması

Önceki BLOCKED satırları public çok-grup API kapsamına aitti. Kullanıcının açık
istemine göre iki farklı mevcut kovaryans yolu artık ayrı ve seçime bağlıdır:

- **Wald:** `core.irt_dif.wald_dif`, mevcut eşzamanlı M1 (1PL, Δb) / M2 (2PL,
  Δa/Δb). Referans N(0,1), odak μ/σ serbest, sabit çapalar; `_contrast_cov` tam
  CΣCᵀ, çapraz grup ve a–b kovaryansları dahil. Mevcut gözlenen bilgi kullanılır;
  yeni Hessian/kalibrasyon motoru yazılmadı. Bu açık Wald seçeneği, eski motoru
  diğer ekranlarda sessizce birincil yapmaz; public IRT-LR kapısı değişmez.
- **Lord:** ayrı public `fit_python_mirt` ML kalibrasyonları/Oakes; 1PL sabit a=1
  ve A=1, B=ortalama farkı. 2PL A=sd(b_R,çapa)/sd(b_F,çapa),
  B=mean(b_R,çapa)−A mean(b_F,çapa). θ_R=Aθ_F+B, a_F*=a_F/A, b_F*=A b_F+B.
  Tam analitik J_R/J_F bütün çapa türevlerini ve çapa–test kovaryanslarını içerir;
  ΣΔ=J_RΣ_RJ_Rᵀ+J_FΣ_FJ_Fᵀ. Grup bağımsızlığı yalnız ayrı Lord örneklemleri
  içindir; eşzamanlı Wald’da çapraz grup terimleri korunur.

Her iki yöntemde yakınsama, sınır dışı serbest parametreler, pozitif tanımlı
serbest/kontrast kovaryansı ve κ≤1e8 zorunlu. 3PL/4PL reddedilir; geçersiz uyumda
χ²/p=NA ve gerekçe. En az iki benzersiz/değişken çapa ve ayrı test ailesi gerekir.
Yeni yöntemler varsayılan otomatik seçime eklenmez. Model seçimi fingerprint’e
katılır. p düzeltmesi önceden seçilen tam yöntem-içi aileye uygulanır; kontrast
kovaryans tabloları UI, Excel ve ZIP’te aynı kaynaklardan üretilir.

Kanıt: `validation/dif_wald_lord/PROTOCOL.md` ve `results/`. 3 sabit fikstürde
**3 mevcut eşzamanlı + 6 ayrı public Python kalibrasyonu** gerçekten çalıştı;
hepsi yakınsadı ve kovaryans kapılarını geçti. Tekrar kullanılabilir NPZ’ler
parametre/kovaryansları kayıpsız saklar; güncellemelerde kalibrasyon tekrar edilmez.
`dif1pl_uniform` eksik veri nedeniyle açık, eşit complete-case kohorta (741/1600)
indirgenmiştir; 859 çıkarım kaydedilir. Diğer iki fikstür 2000’er satırdır.
Üretimde sessiz listwise/imputation eklenmedi. Lord’un iki sabit-N(0,1) modeli ve
Wald’ın serbest-odak-dağılım modeli aynı test değildir; sonuçları eşitlemek amaçlanmaz.

Bağımsız R karşılaştırması yeni hedefli CI işinde, R mirt1.48 ve aynı üç kohortta
çalışır: `Rscript scripts/mirt_irt_b10_reference.R --wald-lord validation/dif_wald_lord/results`.
Eski B10 varsayılan betiği/ham referansları değişmez. Lord’da aynı GH61 ve kütleler;
Wald referansında mevcut mirt-grid61; R kovaryansı a,d→a,b dönüştürülür. Lord R
Jacobiani bağımsız `numDeriv` ile hesaplanır. Kullanıcının **χ² göreli fark ≤1e−3**
eşiği değişmez. Parametre/kovaryans farkları ayrıca raporlanır; farklı parametrizasyonda
ham a,d Wald’ı fiziksel a,b Wald’ıyla aynıymış gibi karşılaştırılmaz. Son gerçek
R yürütme/karşılaştırma durumu CI artifact `comparison.json` ve PR raporundadır;
R çıktısı yoksa VALIDATED veya CERTIFIED yazılmaz. Genel DIF-FWER sertifikası yok.

### Gerçek bağımsız R sonucu (kalibrasyonlar tekrarlanmadı)

R `mirt` **1.48**, workflow `41445711328`, kod `3a2de19`: **9/9 uyum yakınsamış,
sınır/pozitif tanımlılık/κ≤1e8 kontrolleri başarılı**. Kaynak hash’leri, kullanılan
kohortlar, parametre/kovaryans farkları ve tam karşılaştırma özeti
`validation/DIF_WALD_LORD_REFERENCE.json`; ham R/Python CSV, RDS ve stdout/stderr
aynı workflow artifact’ındadır. Aşağıdaki göreli χ² farkları önceden verilen
1e−3 sınırının altındadır:

| Fikstür | Wald max göreli χ² | Lord max göreli χ² |
|---|---|---|
| dif1pl_uniform (741 tam satır) | 6.14815e−4 | 9.69795e−7 |
| dif2pl_mixed (2000) | 1.06847e−5 | 5.76007e−6 |
| dif2pl_none (2000) | 7.40570e−4 | 2.24348e−4 |

En büyük mutlak kontrast parametre farkı Wald **1.53336e−4**, Lord **1.21082e−5**;
kontrast kovaryansı farkı Wald **7.22657e−6**, Lord **2.26991e−6**. Diğer parametre
ve kovaryans farkları ayrıca kaydedilir; bunlar için sonradan kabul eşiği
uydurulmadı. Küçük değere bölünen göreli ikincil farklar χ² eşiğiyle karıştırılmaz
(ör. null Wald kontrastında göreli parametre farkı 1.48021e−3; ayrı Lord
kalibrasyonu kovaryansında bazı küçük elemanlarda göreli fark ≈0.00680).

**R karşılaştırması bu üç kayıtlı tasarımın sayısal χ²/parametre/kovaryans uyumudur; genel CERTIFIED değildir. 1PL Lord için eşit örtük varyans varsayımını doğrulamaz ve çıkarımsal p/q kullanımına izin vermez.** 2PL Lord ve Wald kapsamları korunur.
Bu salt kanıt/durum güncellemesi yeniden R kalibrasyonu gerektirmez; CI kaynak
hash’leri ve aynı önceden tanımlı 1e−3 sınırıyla kayıtlı gerçek kanıtı doğrular.

PR path filtreleri toplam PR farkını da kapsadığından, yalnız kanıt commit’i bile
R işini yeniden tetikleyebilir. `use_reference.py` bu durumda gerçek tamamlanmış
R yürütmesini **yalnız** beş kalibrasyon/referans kaynağı, üç ham veri dosyası ve
18 Python uyum/kovaryans/ayar dosyasının SHA-256 kimliği bire bir eşleşirse kullanır.
Önceden tanımlı 1e−3 sınırı ve bütün yakınsama/sınır sonuçları yeniden doğrulanır.
Herhangi bir kaynak/veri/uyum değişirse gerçek R işi gerekir; eski kanıt sessizce
geçerli sayılmaz. Bu korumanın değişmiş kaynak/NPZ karşı örnekleri geçti. Yeni
R kurulumu veya dokuz kalibrasyonun tekrarı gerekmez; `new_R_fits_this_run=0`
ve asıl yürütme commit/workflow bağlantısı artifact’ta açıkça kaydedilir.

### Son tam regresyondaki arşiv kimliği düzeltmesi

`3a2de19` tam regresyon: **1278 passed, 15 skipped, 2 failed** (1443.68 s).
İki hata aynı kaynak kimliğindeydi: eski GRM kayıtları `core/irt_dif.py` dosyasının
tamamını eski SHA ile karşılaştırıyordu. Kanıt: dosyanın eski **22312 byte’ı**
**aynen korunur**, SHA-256 `f6fcf8bbae297a83e8b3459fc8117f19ec45e5364634a721d5cb4f1551091449`;
tek fark sonuna eklenen `_checked_covariance`, `_contrast_cov`, `wald_dif` tanımları.

Arşiv/manifest/reference hash’leri değiştirilmedi. Yeni doğrulayıcı sadece bu
bilinen dosya/ön-ek için byte kimliğini ve üç yeni, eski adları gölgelemeyen,
dekoratör/default-call/top-level yan etkisi olmayan fonksiyon tanımını kabul eder.
Başka kaynaklar ve tüm ham kayıtlar halen tam-dosya SHA ile kontrol edilir.
Dondurulmuş bağımsız denetleyici değiştirilmeden, yalnız bu kanıtlanmış kaynak
byte’larının hash’ine kapsamlı olmayan tarihsel görünüm verilir; veri/diğer kaynak
hash’leri etkilenmez. Bozuk eski ön-ek, değişmiş veri, eski fonksiyonu yeniden
tanımlama veya top-level tolerans değişikliği reddedilir. İki eski başarısız test
ve iki yeni negatif koruma testi geçti. Lord/Wald kalibrasyon/referans kaynakları
R yürütmesindeki byte’larla aynıdır; tekrar uyum veya ikinci tam süit gerekmez.

## Historical PR130 fix — fixed-SD 1PL Lord is descriptive only

The separate fixed-a 1PL fits force sigma=1 in each group; A=1/B location linking
cannot establish equal true latent SDs. Agreement with R under the SAME imposed
scale does not independently validate this assumption. No cutoff on a fitted
Wald sigma_F (including sigma_F close to 1) is used as evidence.

The shared production/report policy therefore always withholds legacy fixed-SD 1PL Lord p/q,
raw/adjusted aliases and inferential decisions. It preserves chi-square, delta-b,
SE and link/covariance numerics, while marking chi-square explicitly descriptive:
`DESCRIPTIVE / EQUAL LATENT SCALE ASSUMPTION NOT VERIFIED`.

> 1PL Lord testi, grupların örtük yetenek ölçeklerinin eşitliği varsayımına duyarlıdır. Bu varsayım doğrulanmadığından p değeri raporlanmamaktadır. Çıkarımsal DIF değerlendirmesi için Wald DIF veya uygun 2PL Lord analizini kullanınız.

The same policy applies at the center boundary, to existing cached UI results,
and defensively inside CSV/Excel/ZIP/Markdown/plot exporters. It offers no manual
verification toggle. The numerical calibration/reference kernel and frozen R
outputs remain unchanged for reproducibility; their historical numerical tails
are not user-facing inferential results. 2PL Lord and both Wald paths retain
all previous values and gates. No new calibration, Monte Carlo or full local
regression was run. Targeted tests use saved real fits and cover sigma values,
all p/q aliases, legacy exports, exact descriptive numerics, UI and unaffected
2PL/Wald results. Existing R/source/archive integrity tests remain strict.

Original raw R coefficient/covariance CSVs are retained in workflow `41445711328`,
artifact `13142108086` (SHA-256 `7f779513508bb37b9e6d78ec83434d58414b63640ad0a64ec267a50a7805c1d6`):
`dif1pl_uniform/r_single_{R,F}_raw_parameters.csv` and
`dif1pl_uniform/r_single_{R,F}_raw_vcov.csv`. They are not rewritten or regenerated
by this fix; no assumption-verification claim is inferred from them.


## Free-SD Rasch extension — independently gated

The new `core.dif_lord_rasch` path reuses `core.rasch.fit_rasch`, with a=1,
mean=0 and independent free SDs. Common units follow from a=1, and anchor
centering supplies location; no equality of latent SDs is assumed. Covariance
is the b block of the FULL inverse information in (b,log SD), with complete
anchor Jacobians. Existing PD/condition/bound gates and independent GH101
audit apply. The old numerical kernel and original R artifacts are immutable.

[Preregistered protocol](lord_rasch_free_scale/PROTOCOL.md): actual R mirt1.48
Rasch/GH61 and101, same741 complete-case cohort, independent numerical delta
Jacobian, pre-existing relative chi-square limit1e-3. No reference PASS is
claimed until actual execution. Source/data-identity checking fails closed;
legacy cached fixed-SD results never acquire inference. New results retain
conditional inference and NOT FWER CERTIFIED wording even after numeric PASS.

[Sequential GRM-FWER protocol](grm_fwer_sequential/protocol.json) is a separate
scoped experiment, not evidence for Rasch Lord/Wald/SIBTEST/Raju or selected
anchors. Fixed500/1000/1500 looks, two impacts, twelve directional CP endpoints
share alpha.05. Liberal control<=.075 differs from nominal .05; both control
and inflation flags can coexist. Failed fits count as operational events but
cannot manufacture evidence of inferential Type-I inflation. Old30 pairs are
not reused because environment identities differ. No interim certification.
