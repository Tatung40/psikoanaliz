# MTK özellik ve kanıt denetimi — 2026-11-02

Taban: `511bc1d` (PR #127 dahil). Bu rapor genel bilimsel sertifika değildir.
**PASS** yalnız belirtilen işlev/cebir kontrolünün mevcut olduğunu belirtir;
**PARTIAL** sınırlı uygulama veya koşullu kanıt; **UNVERIFIED** eşleşen doğrulama
kanıtı bu incelemede denetlenmedi; **UNSUPPORTED** bilerek kapalı;
**MISSING** güvenli uygulaması olmayan gerekli özellik anlamındadır.
Kestirim başarısı, sayısal test, bağımsız benchmark, bilimsel geçerlilik ve UI
entegrasyonu ayrı değerlendirilir. Test başarısından sertifika türetilmez.

## Model × özellik matrisi

SH/GA = parametre standart hatası/güven aralığı; kişi SH, posterior SD ve
`1/sqrt(TIF)` farklıdır. Yerel motorlar otomatik kütüphane alternatifi değildir.

| Model / gerçek motor | Kestirim, tanımlama | SH/GA | Kişi | Uyum | DIF çıkarımı | Eksik veri | Eğri/bilgi/SEM | Wright | UI/indirme | Bağımsız benchmark / bilimsel düzey |
|---|---|---|---|---|---|---|---|---|---|---|
| Yerel Rasch `core.rasch`, a=1, σ serbest | PASS | PARTIAL | PASS | PARTIAL | PARTIAL | PARTIAL | PASS | PASS | PASS | Kaydedilmiş R Rasch Δparam=1.276e−6; koşullu, genel GA sertifikası yok |
| Python Rasch / sabit-a 1PL, σ=1 | PASS | PARTIAL | PASS | PARTIAL | PARTIAL | UNSUPPORTED adaptörde | PASS | PASS Rasch yolu | PASS | R 1PL Δparam=.0004417; p6 normal θ, N1000/J12 koşullu |
| Python ortak-serbest-a 1PL, σ=1 | PASS | UNVERIFIED kapsam | PARTIAL | PARTIAL; S-X² p kapalı | UNSUPPORTED bu spesifikasyon | UNSUPPORTED adaptörde | PASS | PARTIAL betimsel | PASS | UNVERIFIED: sabit-a kanıtı aktarılamaz |
| Python 2PL | PASS | PARTIAL | PARTIAL | PARTIAL | PARTIAL | UNSUPPORTED adaptörde | PASS | PARTIAL betimsel | PASS | R Δparam≤.000923; p6 kapsama .945–.951, koşullu |
| Python 3PL ML/MAP | PASS | PARTIAL MAP; ML GA kapalı | PARTIAL | PARTIAL; S-X² p kapalı | UNSUPPORTED bootstrap çıkarımı | UNSUPPORTED adaptörde | PASS | PARTIAL betimsel b | PASS | Eşlenmiş ML Δparam=.0002353, ΔLL=1.537e−7; MAP kanıtı ayrı |
| Python 4PL ML | PASS uygulama | UNSUPPORTED GA | PARTIAL | PARTIAL; S-X² p kapalı | UNSUPPORTED bootstrap çıkarımı | UNSUPPORTED adaptörde | PASS | PARTIAL betimsel b | PASS | PARTIAL NUMERICAL AGREEMENT / NOT CONVERGED / NOT CERTIFIED |
| Python 4PL MAP | PASS uygulama | UNSUPPORTED GA | PARTIAL | PARTIAL; S-X² p kapalı | UNSUPPORTED bootstrap çıkarımı | UNSUPPORTED adaptörde | PASS | PARTIAL betimsel b | PASS | PARTIAL: p1 normal θ, N3000/10000 × J12/30 nokta kestirimi; bağımsız R MAP UNVERIFIED |
| Python GRM | PASS | PARTIAL | PARTIAL | PARTIAL; S-X² kapalı | PARTIAL sabit çapa | UNSUPPORTED adaptörde | PASS CRC | UNSUPPORTED bu UI | PASS | p4 koşullu; R eşleme altyapısı mevcut, yeni R çalışması yok |
| Python PCM | PASS a=1 | PARTIAL | PARTIAL | PARTIAL; S-X² kapalı | PARTIAL sabit çapa | UNSUPPORTED adaptörde | PASS CRC | UNSUPPORTED bu UI | PASS | p4 N2000 adım kapsama .940; R gpcm a=1 eşleme altyapısı |
| Python GPCM | PASS | PARTIAL | PARTIAL | PARTIAL; S-X² kapalı | PARTIAL sabit çapa | UNSUPPORTED adaptörde | PASS CRC | UNSUPPORTED bu UI | PASS | p4 N2000 a/adım kapsama .952/.966; R kümülatif adım dönüşümü |
| Yerel RSM `core.irt_rsm` | PASS, Στ=0; σ seçilebilir | PARTIAL | PASS EAP/WLE | PARTIAL RSM–PCM LR | PARTIAL sabit çapa | PARTIAL MAR varsayımı | PASS CRC | UNSUPPORTED bu sayfa | PASS | p3 koşullu; N300/J20 δ kapsama .915 başarısız; public fit_mirt RSM UNSUPPORTED |
| Python NRM | PASS, kategori0 A=D=0 | PARTIAL açık Oakes | PARTIAL | UNSUPPORTED ordinal madde uyumu | PARTIAL yerel sabit çapa; public çok-grup p kapalı | UNSUPPORTED adaptörde | PASS nominal CRC | UNSUPPORTED | PASS | p4 koşullu; nominal yön/yansıtma şart; R nominal eşleme altyapısı |

`PASS` UI/indirme, mevcut model yollarının bağlandığı anlamındadır; her formatın
her sayfada bulunduğu anlamına gelmez. Python ortak rapor CSV, kişi XLSX, ikili
Excel/ZIP/PNG ve B14/RSM CSV/XLSX/ZIP yolları modele göre değişir. TXT/bağımsız
R kod üretimi zorunlu değildir; desteklenmeyen format eklenmedi. Kategori etiketleri,
ID ve bilimsel notlar ilgili raporlarda korunur. Nominal kategori sırası başarı değildir.

## Kaynak ve sınırlar

- `core/irt_python_mirt_adapter.py`: gerçek `fit_mirt` API; EM, ML (3PL/4PL MAP
  seçilebilir), varsayılan GH21/max_iter500/tol1e−4; ortak a yalnız açık 2PL eşitlik
  kısıtı. NRM için açık `se_method='oakes'`; eksik/nonfinite/boş kategoriler reddedilir.
- `validation/SCIENTIFIC_READINESS_2026-10-31.md`,
  `validation/POLYTOMOUS_MC_VALIDATION.md`, `validation/RSM_VALIDATION.md`:
  mevcut koşullu kurtarma/kapsama kanıtı; tarihsel kümeli MCSE eksiklikleri korunur.
- `validation/r_mirt_phase1/final_summary.json` ve
  `validation/r_mirt_phase1/observed_information/report.json`: kaydedilmiş R kanıtı;
  4PL yakınsamama, zayıf pratik tanımlanma ve optimizer sınırlılıkları sürer.
- `validation/r_mirt_polytomous/README.md`: GRM d=−ab, PCM/GPCM kümülatif adım,
  NRM referans/yansıtma eşlemeleri; Python GH61 ile R eşit aralıklı101 ızgara
  aynı kestirim quadrature'ı değildir. Altyapının varlığı yeni bağımsız benchmark
  veya genel sertifika anlamına gelmez. Açık #65/#69/#72/#74 dalları değiştirilmedi.
- `core/irt_person_abilities.py`: genel ortak ekranın EAP'si kesik [-4,4]
  ızgarasında; Rasch GH61 yolu ayrı. Kuyruk hassasiyeti ve MAP/ML kişi puanları
  **UNVERIFIED**; yeni skor yöntemi eklenmedi.
- `core/python_mirt_dif.py`: public çok-grup LR betimsel, p/q/Wald kapalı;
  Hessian/SE API eksikliği bağımsız doğrulama gibi sunulamaz. Yerel sabit-çapa
  modüllerinin modele özgü kanıtı bu API'ye aktarılmaz.
- PR #123–127 arşivleri ve eşikleri değişmedi:
  **NUMERICALLY VALIDATED — 30 PAIRED SEEDS**; **FWER NOT CERTIFIED**.
  Bu teknik sonuç genel FWER kontrolü değildir. Yeni 100/500/2500 deneyi yapılmadı.

## Doğrulanan ve giderilen kusurlar

1. **P0, RSM eski sonuç:** `rsm_request` önce eski sütun/ayarlarla tekrar okunuyordu;
   madde, latent ölçek, GH veya LR seçimi değişince önceki tablo/grafik/indirmeler
   kalabiliyordu. Güncel seçimlerin tümü kayıtla karşılaştırılır; veri içerik hash'i
   korunur. Uyumsuzlukta sonuç render edilmeden istek silinir.
2. **P1, sıfır bilgi:** B14 `max(TIF,1e−300)` ile TIF=0 için SEM=1e150 üretiyordu.
   Sıfır/negatif toplam bilgi artık tanımsız (NaN); pozitif bilgide 1/√TIF aynen kalır.
3. **P0, kanıt kapsamı:** 3PL/4PL kullanım etiketi N ve J'ye göre sınırsız
   ekstrapolasyon yapıyordu. `supported` yalnız çalışılmış normal-θ N/J hücrelerinde;
   diğer büyük tasarımlar `caution`. Eşik, önsel veya kestirim değişmedi; bu etiket
   normal θ varsayımını ya da bu veri için bilimsel yeterliliği kanıtlamaz.
4. **P0, S-X² kapı atlama:** Python model ekranı, ikili rapor paketi ve B14 tanıları
   doğrudan `mirt.itemfit` çağırarak mevcut yerel-uyum kısıtlarını uygulamıyordu.
   Ortak `library_itemfit` artık yakınsama ister; NRM/RSM'yi reddeder; ordinal
   ailelerde yalnız betimsel Infit/Outfit ister. 3PL/4PL ve ortak-serbest-a 1PL
   için S-X² istatistiği betimsel tutulur, p/ayarlanmış-p bastırılır. UI, CSV,
   Excel/ZIP aynı tablodan üretilir; sabit-a 1PL/2PL de NOT CERTIFIED notu taşır.

## Yeni çalıştırılan kontroller ve durma kararı

- Önceki kodla 7 no-fit regresyonda **5 hata/2 başarı**: dört RSM ayar değişimi
  eski sonucu tutuyor, sıfır bilgi SEM'i sonlu çıkıyordu. Aynı-boyut veri değişimi
  ve değişmeyen giriş davranışı zaten doğruydu; korundu.
- Yeni no-fit kontroller: tüm RSM analiz ayarları, aynı-boyut veri değişimi,
  değişmeyen girdiler, sıfır bilgi, tasarım kapsamı ve S-X² UI/CSV/Excel kapıları.
- Mevcut küçük RSM AppTest ve dört ailenin bağımsız sonlu-fark Fisher bilgisi
  kontrolü çalıştırıldı; yeni simülasyon yok. Mevcut sabit-parametre S-X² referans
  testi çalıştırıldı. Bunlar yeni bağımsız R kalibrasyonu değildir.
- Hedefli doğrulama: **35 passed** (12 + 6 + 11 + 6); bunlara üç gerçek
  motorlu 3PL/GRM/NRM Streamlit akışı ve mevcut CSV/XLSX/ZIP güvenlik testleri dahildir.
- Son tam regresyon CI'da bir kez istenir: PR gövdesindeki
  `<!-- full-regression -->` işareti mevcut pytest işinin tam-süit yolunu seçer;
  normal PR'lerin seçili test yolu değişmez. Kalıcı dal adına özel koşul eklenmedi.
- **BLOCKED/research:** genel 4PL ML sertifikası, 4PL GA, public çok-grup p/Wald,
  ordinal S-X² çıkarımı, NRM uyum/Wright, genel MAR/MNAR ve DIF-FWER sertifikası.
  Bunlar yeni/bağımsız bilimsel kanıt gerektirir; 0-TL hızlı düzeltmeyle üretilmiş
  sayılmaz. P2: kesik kişi ızgarası hassasiyeti ve ek formatlar. Otomatik merge yok.

## Tam regresyon sonucu ve envanter düzeltmesi

CI `41398075970`, `c187ee8`: **1234 passed, 15 skipped, 1 failed**, 1461.33 s.
Tek hata `test_inventory_is_current_and_includes_missing_data_page`: yeni ortak
`library_itemfit` importu üretilmiş kod envanterinde yoktu. Mevcut
`tools/inventory_reproducible.py` ile yeniden üretim tam **bir satır** ekledi;
başarısız test **1 passed**. Üretim kodu değişmedi. Toplam hedefli başarı 36.
Tam süit ikinci kez çalıştırılmaz; PR gövdesindeki opt-in işareti kaldırılarak
son commit için mevcut zorunlu CI seçili regresyonları izlenir. İlk başarısız
CI kaydı korunur; tam süit hatasızmış gibi gösterilmez.
