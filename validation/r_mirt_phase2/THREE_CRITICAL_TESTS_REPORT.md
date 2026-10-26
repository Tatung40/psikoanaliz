# Üç kritik doğrulama — PR #75

26 Ekim 2026. Dal: validation-r-mirt-phase2-nine-models; başlangıç b053886. Python 3.11.17 / Python mirt 1.2.0. Bu sonuçlar işlevsel yazılım doğruluğudur; bilimsel model sertifikasyonu değildir.

| Test | Sonuç | Kanıt |
|---|---|---|
| max_iter / tol / n_quadpts | PASS | Gerçek EM: 1/3 iterasyon sınırı, toleransla 2/19 iterasyon; gerçek GH11/GH31 düğüm/ağırlıkları |
| Rasch Wright Map | PASS | Gerçek LSAT6 kalibrasyonu; EAP, b, histogram, CSV/Excel kimliği; AppTest ve Plotly grafik değerleri |
| Dokuz model regresyonu | PASS | Dokuz adlandırılmış gerçek kalibrasyon ve desteklenen çıktılar; tam regresyon 947 geçti, 2 atlandı |

## Çözücü

Kurulu mirt.fit_mirt gerçek imzası max_iter=500, tol=1e-4, n_quadpts=21 destekler. İki UI girişi → mevcut servis (kullanıldığı yerde) → adaptör → gerçek motor aktarımı tamamlandı. Uygulanan ayarlar sonuçta saklanır; değişiklikler önceki çıktıların gösterilmesini engeller. Bilinmeyen seçenekler, bool/kesirli iterasyon/grid ve sıfır/negatif/NaN/sonsuz tolerans açıkça reddedilir.

Gerçek EMEstimator.fit tamamlandıktan sonra kullanılan grid ve yakınsama geçmişi gözlendi; sahte fit çıktısı yoktur. Aynı veride iterasyonlar [1,3,2,19]: ilk ikisi kasıtlı sınırda yakınsamadı; tol=1e6 iki iterasyonda, tol=1e-8 19 iterasyonda durdu. Son sıkı amaç farkı 2.8380782168824226e-9. Gevşek tolerans yalnızca kontrolün etkisini ölçer; bilimsel kabul eşiği değildir. ML altında marjinal logLik, MAP altında logposterior değişimi kullanılır.

GH11/GH31 düğüm/ağırlıkları ve normal momentleri doğrulandı. Sınırda ek nihai E-değerlendirmesi geçmişe yazılabilir; bu ek M-adımı değildir. Varsayılan adaptör ve doğrudan kütüphane 1PL fit'leri katsayı, LL, iterasyon ve yakınsama bakımından aynıdır. Her iki UI'da seçilen ayarlar gerçek fit'e ulaştı.

## Gerçek Rasch verisi ve Wright Map

Kontrollü mevcut fikstür: 1.000 kişi × 10 madde, 13 iterasyon; bağımsız posterior θ farkı en çok 1.7763568394002505e-15. Bu önceki simülasyon fikstürüdür; saha verisi olarak sunulmaz.

Ek gerçek kamu verisi, mevcut scripts/real_lsat6_ctt_pilot.py kaynağı [psych::lsat6](https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/psych/lsat6.csv): 1.000 kişi × 5 madde; SHA256 2912afd22a32770c3eebd55172f508c187aac0b728ac60adcf175d63c3e4c445. Bir gerçek Rasch fit'i 12 iterasyonda yakınsadı; LL=-2473.053851600649. Mevcut kaynak marjinal oranları kontrol edildi.

Madde difficulty katsayısı fiziksel b'dir: gerçek fit'te her madde için P(θ=b)=0.5 doğrulandı. EAP ve posterior SD, desteklenen mirt.fscores(method="EAP", n_quadpts=61) API'sindendir. Kişi ve madde ölçeği sabit N(0,1), D=1; yeniden merkezleme yoktur. Kalibrasyon GH21, puanlama GH61 kullanır; latent ölçek aynıdır.

LSAT6 bağımsız posterior farkları: θ 4.440892098500626e-16, SD 9.992007221626409e-16. θ aralığı [-2.03752614178942,0.7078449533064901]. Üç sıfır-tam ve 298 tam-doğru örüntüde EAP sonludur. Madde sırası Q1,Q5,Q4,Q2,Q3; histogram toplamı 1.000. CSV/Excel–grafik kimliği bellekte doğrulandı. Gerçek yanıt/kişi satırları kaydedilmedi; yalnızca toplulaştırılmış grafik/histogram ve madde konumları saklandı.

Kontrollü testler ayrıca yinelenen dataframe indekslerinde ID sırası, ortak eksen, gerçek histogram, eksik veri/geçersiz kimlik/yakınsamama ve değişen ayarda grafik/indirme bastırılmasını doğrular. Mevcut Rasch grafik şablonu yeniden kullanıldı; WLE etiketi gerçek yöntem EAP olarak düzeltildi.

Yerel piksel yakalama BLOCKED: Chromium Crashpad soket kısıtı / render zaman aşımı. Yerelde başarılı piksel testi iddia edilmez. Mevcut CI Chromium ortamı kontrollü ve LSAT6 grafiklerini sıfır yeni fit ile ayrıca kontrol eder: gerçek DOM koordinatları, etiketler, sayılar, ortak eksen ve PNG'ler. Son commit'e bağlı sonuç ve görüntüler [PR #75 Checks](https://github.com/Tatung40/psikoanaliz/pull/75/checks) ve PR kabul kaydındadır; hata CI'ı başarısız yapar.

## Dokuz yol ve bilimsel kapsam

Rasch, 1PL, 2PL, 3PL, 4PL, GRM, PCM, GPCM, NRM gerçek mevcut fikstürlerle yakınsadı; iterasyonlar 13,13,20,14,14,15,21,45,35. Parametre sayıları 6,6,12,18,24,24,18,24,36. Parametre/LL/IC, olasılıklar, bağımsız sonlu-fark IIF, TIF/SEM, EAP/SD, ID sırası, CSV/Excel ve yakınsamama bayrakları kontrol edildi. En büyük bilgi kimliği farkı 9.629586017467773e-11; mevcut türev toleransları gevşetilmedi.

Bu dalda Python Rasch ve 1PL aynı a=1, N(0,1) parametrizasyonudur: dokuz etiket, sekiz farklı kütüphane ailesi. 1PL'ye sessizce ortak serbest eğim eklenmedi. Yerel Rasch'ın serbest latent SD tanımı ve servis koruması aynen kaldı. RSM dokuz yolun yerine sayılmadı. Wright yalnızca Python Rasch'a eklendi; diğer modellerin davranışı korunur.

3PL/4PL odaklı regresyon mevcut belgelenmiş MAP önsellerini kullanır; mevcut ML testleri/uyarıları korunur. Bu, bağımsız 4PL ML yakınsama/sertifikasyon sorunlarını çözmez. Algoritma, model varsayılanları, önseller ve bilimsel/inferans eşikleri değişmedi. Yeni R fit'i gerekmedi/çalıştırılmadı: hesaplama değişmedi, varsayılan aynı-fit kimliği doğrulandı; mevcut R kanıtları korunur. Genel model uygunluğu veya SE/CI kapsam sertifikasyonu verilmez.

## Test, commit ve CI kayıtları

- Yeni hedefli testler: çözücü 15 + ayrı legacy UI 1; Wright 3; dokuz yol 9 (toplam 28).
- Etkilenen mevcut adaptör/servis/sidebar/binary UI hedefli kümesi: 40 geçti.
- Gerçek LSAT6 ek kontrolü: PASS, tek fit; yeni simülasyon yok.
- Tek tam regresyon: 947 geçti, 2 atlandı, 199 uyarı; 840.08 s. Mevcut atlamalar: isteğe bağlı raschpy yok ve opt-in yavaş 4PL refit kapalı; yeni üç kontrol atlanmadı.
- Uygulama/test kaynak commit'i: 9de589cc9ffbff98c5e9e4ec06282bf89943f87e. Sonraki 3ef86770f4f21c730f468eea8dbf363b5562ba88 yalnızca ek gerçek-veri/browse doğrulama betikleri ve kanıtıdır; uygulama kodu değişmez.
- Makine kanıtları/kaynak hash'leri: [three_checks.json](evidence/three_checks.json), solver.xml, wright.xml, nine_models.xml, lsat6_summary.json ve grafik/tablo çıktıları.
- [PR #75](https://github.com/Tatung40/psikoanaliz/pull/75), [commit'e bağlı CI sonuçları](https://github.com/Tatung40/psikoanaliz/pull/75/checks), [PsikoAnaliz tests](https://github.com/Tatung40/psikoanaliz/actions/workflows/pytest.yml), [MTK scientific gates / Wright](https://github.com/Tatung40/psikoanaliz/actions/workflows/mtk-scientific-gates.yml). Son koşu ID'leri ve gerçek sonuçlar PR kabul kaydına eklenir; belge henüz çalışmamış CI için başarı iddia etmez.

Yeniden üretim (aynı ortam/bağımlılıklarla):

    PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 python -m pytest -q tests/test_phase2_solver.py tests/test_phase2_wright.py tests/test_phase2_nine_models.py
    # Yalnızca gerçek veri kanıtını tekrar üretmek gerekirse (bir fit):
    PYTHONPATH=. python validation/r_mirt_phase2/check_real_rasch.py --output /tmp/real-rasch
    # Mevcut CI Playwright ortamında, sıfır yeni fit:
    PYTHONPATH=. python validation/r_mirt_phase2/check_wright_browser.py --output /tmp/phase2-wright-browser
    # Tek tam regresyon:
    PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 python -m pytest -q

Kalan sınırlar: yerel piksel ortamı; Rasch/1PL alias kapsamı; eksik veride açık ret; işlevsel PASS'tan bilimsel/klinik sertifikasyon çıkarılmaması. main veya diğer dallar değiştirilmez; merge/deploy yapılmaz.
