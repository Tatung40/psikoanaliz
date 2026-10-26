# PR #73 + #75 → main entegrasyon kaydı

Bu dosya değişmez kaynak/yerel kanıt kontrol noktasıdır. Son entegrasyon commit'inin CI koşuları, gerçek main birleştirme SHA'sı ve dağıtım sonucu ilgili entegrasyon PR'ının kabul kaydında tutulur. Henüz çalışmamış CI veya canlı dağıtım için PASS üretilmez.

| Aşama | Kayıt |
|---|---|
| 1 — #75 → Phase 1 | PASS: normal GitHub merge, baş SHA ve güncel üç CI doğrulandı |
| 2 — #73 + Phase 1/2 | PASS: mevcut main tabanında iki kaynak birleştirildi; hedefli testler ve gerçek uygulama tarayıcı kontrolü geçti |
| 3 — Regresyon kapısı | Yerel hedefli/Chromium PASS; tam yeşil sonuç kesin entegrasyon başında zorunlu CI kapısıdır |
| 4 — main + canlı doğrulama | main merge/CI kaydı son PR'da; canlı erişim aşağıdaki nedenle doğrulanamadı, üretim başarısı iddia edilmez |

## Değişmez kaynaklar

- [PR #75](https://github.com/Tatung40/psikoanaliz/pull/75), baş efcd9b317a80d9df3500927d2362af48792bcba1; Phase 1 merge: 27443a3dd6b8710fbacfc8d14bfe71d6f4d91f4a. Üç güncel CI başarılı, açık inceleme/çözümlenmemiş yorum yoktu.
- Başlangıç main: ff4aa6a244a9e36873a5666a0812c6d2836437f0.
- [PR #73](https://github.com/Tatung40/psikoanaliz/pull/73): f1e0814b3385f6de463ef8f732628b249bcb8331. Sonraki çözücü/Wright commit'leri de dahil edildi.
- Entegrasyon merge commit'leri: f75a5bb, 46557de; uygulama düzeltmesi [eb54654f458480c27555ca1d6071d93e535095df](https://github.com/Tatung40/psikoanaliz/commit/eb54654f458480c27555ca1d6071d93e535095df); eski mock beklentisi / tam CI kapısı: 8ff60b63eca44956ea65bcaed22e78ce2b9e5d5c.
- Dal: integration/73-75-main. Bu commit'lerin ilişkili son entegrasyon PR'ı, main merge SHA'sı ve koşu bağlantıları için yetkili kabul kaydıdır.

## Korunan davranış ve gerekli çatışma çözümleri

UI → adaptör → kamuya açık mirt.fit_mirt aktarımı tekleştirildi. Phase 2 skaler API ve PR73'ün mevcut sınırları korunan solver_settings sözlük API'si birlikte desteklenir; çelişen özel seçenekler reddedilir. Varsayılanlar 500 / 1e-4 / 21, algoritma, önseller ve inferans eşikleri değişmez. Uygulanan seçenekler özet/CSV'de kaydedilir.

PR73'ün mevcut yedi çıktı grubu, üç grafik seçimi, varsayılan görünümü, isteğe bağlı gelişmiş modu ve isteğe bağlı Rasch Wright kapısı korunur. Bu, tam bir SPSS ürünü iddiası değildir. Çıktı seçimleri fit'i değiştirmez. Model karşılaştırması yalnızca aynı veri/kodlama/çözücü ayarlarında yakınsamış ML fit'lerindedir; MAP katılmaz ve başarısız yeni fit eski satırı korumaz.

Rasch kişi tablosu, Wright ve CSV/Excel, Phase 2'de doğrulanan mirt.fscores GH61 EAP/posterior SD'yi paylaşır. PR73'teki kesilmiş [-4,4] yaklaşımı Rasch için ikinci bir kişi skoru üretmez. Diğer modellerin mevcut kişi kestirimleri değiştirilmedi. Gerçek ID sırası ve yinelenen dataframe indeksleri korunur; ID/ayar/veri değişince eski sonuç gösterilmez. NRM nominaldir: ordinal Infit/Outfit/S-X² seçimi kapalıdır.

Kimliklendirme ayrımı korunur: kütüphanenin varsayılan 1PL API'si ile Python Rasch a=1, N(0,1) kullanır; mevcut arayüz 1PL yolu açık ortak-serbest-a eşitlik kısıtını kullanmaya devam eder. Yerel serbest-SD Rasch ayrı kalır; RSM dokuz yolun yerine sayılmaz. İşlevsel PASS, bağımsız R sertifikasyonu değildir; özellikle 4PL ML sınırlılıkları devam eder.

Orijinal validation/r_mirt_phase1 kanıtları git diff ile değişmemiş doğrulandı; yeni R fit'i veya simülasyon başlatılmadı. Tarihsel Phase 2 raporları kendi kaynak commit'lerine aittir; yeni arayüz parametrizasyonuna sessizce bilimsel onay aktarmak için kullanılmaz.

## Gerçek kontroller

- Çözücü: 24 hedefli test geçti; gerçek 1/3 iterasyon sınırı, 2/19 tolerans etkisi, GH11/GH31 ve eski sözlük API'si.
- Wright: 3 hedefli test geçti; gerçek EAP/b, aynı ölçek, CSV/Excel ve yakınsamama koruması.
- SPSS entegrasyonu: 2 test geçti; çıktı seçimlerinde yeniden kalibrasyon yok, sayısal kimlik, farklı GH ayarlarını karşılaştırmama, geçersiz ayarlarda kontrollü hata.
- Dokuz yol/kişi testleri: 58 geçti + iki düzeltilmiş UI alt testi geçti; sekiz Wright testi diğer aileler için mevcut NOT APPLICABLE atlamalarıdır. Rasch fikstürü doğru model etiketiyle çağrılır; NRM testleri desteklenmeyen ordinal tablo beklemez.
- Gerçek Chromium + gerçek app.py: PASS. Mevcut kontrollü 1000×10 Rasch CSV yükleme, tüm varsayılan SPSS kontrolleri, fit, CSV/Excel, çıktı gizle/göster, gerçek Wright DOM koordinatları ve 1000 kişilik histogram; 700 / 1e-6 / GH31 gerçek yeniden fit; tol=0 hatasında eski grafik/kişi çıktılarının bastırılması. Görüntüler gerçek uygulamadandır, maket değildir.
- Kaynak hash'leri ve yerel tarayıcı kanıtı: [local_browser.json](evidence/integration_73_75/local_browser.json). Büyük PNG/trace dosyaları Git tarihine eklenmez; CI artifact'ında tutulur.

İlk tam regresyon: 1015 geçti, 1 eski mock beklentisi başarısız, 10 mevcut/uygulanamaz atlama; 1027.87 saniye. Tek hata, yönlendirme testinin artık açık aktarılan değişmemiş varsayılanları beklememesiydi. Test dört argümanı ve sonuç ayarlarını tam eşitlikle doğrulayacak şekilde düzeltildi; ilgili iki test geçti. Hesaplama kodu değişmedi. Nihai tam yeşil kanıt için entegrasyon PR'ında mevcut hedefli CI komutunun yerine onun üst kümesi olan tam pytest koşusu zorunlu tutulur; aynı koşu ayrıca ikinci kez hedefli listeyi çalıştırmaz. Diğer formal kapılar ve main tam regresyonu korunur.

CI: [PsikoAnaliz tests](https://github.com/Tatung40/psikoanaliz/actions/workflows/pytest.yml), [MTK scientific gates](https://github.com/Tatung40/psikoanaliz/actions/workflows/mtk-scientific-gates.yml). Kesin baş SHA/koşu ID'leri son PR kabul kaydındadır. Main push'ta mevcut bilimsel ve gerçek tarayıcı kapıları da çalışır; üretime merge ancak gereken baş kontrolleri yeşilken yapılır.

## Dağıtım erişimi

README hedefi main / app.py / https://psikoanaliz.streamlit.app/; .streamlit/config.toml yalnızca headless/theme/telemetri ayarlarını içerir, gerçek Cloud dal bağını kanıtlamaz. Canlı sağlık uç noktası ön denemesi: URLError, Tunnel connection failed: 403 Forbidden (çalışma ortamının ağ geçidi; uygulamanın hata verdiği sonucu çıkarılmaz). Streamlit Cloud yönetim/deployment erişimi yoktur. Ağ kısıtı aşılmaz; bu izin/kanıt olmadan canlı yeni kod, dal/entrypoint bağı veya üretim smoke PASS iddia edilmez. main birleştirmesi ve ana dal CI başarısı üretim başarısından ayrı raporlanır.
