# PCM: uçtan uca durum ve bağımsız R karşılaştırması

Bu belge kapsamı tanımlar; gerçek çalıştırma raporu ve özgün CSV/JSON/log artifact bağlantıları PR açıklamasında tutulur. Test başarısı bilimsel sertifika değildir.

## Mevcut bileşenler ve tamamlanan bağlantılar

| Bileşen | Kaynak / davranış |
|---|---|
| Kestirim / servis | `core.irt_b14.fit_b14(data, "PCM")` → mevcut Python mirt1.2.0 PCM ML; model yeniden yazılmadı |
| Kimliklendirme | a=1 sabit, θ~N(0,1); maddeye özgü serbest eğimli GPCM değildir |
| Adımlar | `fit.model.steps`; kategori/adım indeksleri korunur, adımlar sıralanmaz |
| Kişi kestirimi | `core.irt_b14_persons_info.person_scores`: mevcut EAP ve posterior SS |
| Olasılık / eğri | `core.irt_b14_model_curves.fitted_category_curves`: doğrudan model API'si |
| Bilgi / hata | `information_table`: madde bilgisi, toplam test bilgisi, koşullu SEM=1/√I; parametre SH değildir |
| Uyum / tanılar | `model_diagnostics`: mevcut infit/outfit/S-X² ve model tanıları; kullanılamayan çıktılar açıkça belirtilir, geçerlik iddiası üretilmez |
| Girdi / çıktı | `load_dataset`: CSV/XLSX; mevcut B14 CSV/Excel/ZIP ve Türkçe açıklamalı rapor |
| Arayüz | Python mirt → B14 Çok Kategorili Rapor → PCM; ayrı Python mirt PCM ekranındaki mevcut tablolar ve mobil grafikler korunur |
| Yakınsama | Motor bayrağı ve meta tablosu; yakınsamama belirgin hata olarak gösterilir |

B14 özetine eksik AIC/BIC eklendi. Veri değerleri (boyut aynı kalsa da), madde/model/kategori ayarları değişince sonuç, tanı, model karşılaştırması ve ZIP temizlenir; başarısız yeniden kestirim eski sonucu korumaz. Doğrulanmış indirme davranışı korunur.

## Ortak bağımsız hedef

PR #65'in başarılı altyapısı aynı dosyalarda model seçeneğiyle yeniden kullanılır; yeni bir doğrulama çatısı yoktur. GRM varsayılanı ve mevcut çalışan modeller korunur. Kaynak bağımlılık: PR65 commit03a6250; çalışma dalı güncel main üzerine bu altyapıyı içerir, PR65 dalı değiştirilmez.

- Aynı mevcut `poly_ordinal.csv`, 500 kişi ×8 madde, her maddede kategoriler 0/1/2/3. Bu GRM kökenli fixture üzerinde PCM uygulamak sayısal motor karşılaştırmasıdır; PCM gerçek-parametre kurtarma çalışması değildir.
- N(0,1), D=1, 61 GH düğümü/ağırlığı, ML/önselsiz, aynı önceden verilmiş adımlar; ilk maddede [.5,-.5,.2] sırasız adım tanığı korunur.
- Python PCM: sabit a=1, serbest 24 adım, adımlar [-6,6].
- Bağımsız R mirt1.48 `gpcm`: a1=1; ak0..ak3=0..3; d0=0; yalnız d1..d3 serbest. `d_s=-Σ_{t≤s}b_t`, `b_s=-(d_s-d_{s-1})`. Bunlar GRM eşikleri değildir.
- R eşitsizlikleri adım farklarını tam sınırlar; bağımsız d kutuları PCM adım sınırlarının yerine kullanılmaz. |d_s|≤18 yalnızca üç adım için gereksiz geniş dış kutudur.
- Aynı EM tavanı3000, tol1e-8, hızlandırma yok; durma tanımları farklıdır ve raporlanır. Kısıt ihlali, yakınsamama veya eksik yürütme başarıya çevrilmez.

R ve Python birer kez çalışır. Yerel tekrar işaretçisi aynı denemeyi reddeder. Mevcut tek GitHub Actions iş akışının `pcm-fit` modu kullanılır; bu dalın PR kontrolü PCM çalıştırır, GRM yeniden kalibre edilmez. GRM/PCM no-fit preflight birlikte korunur.

## Kaydedilen karşılaştırmalar

`parameter_comparison.csv`: her madde/adım için R değeri, Python değeri, mutlak fark, |R| paydalı göreli fark ve serbest/sabit etiketi; R=0 için göreli fark tanımsız/boştur. Özgün parametreler, kategori eğrileri, bilgi, meta ve loglar değiştirilmeden saklanır.

JSON ayrıca ortak GH61 kişi marjinal logLik, kategori ve kümülatif eğriler, madde/test bilgisi, AIC/BIC tanım denetimleri, fiziksel/native gradyanlar, yakınsama/sınır tanıları, sürümler ve girdi/kaynak hash'lerini içerir. PCM ortak-logLik ve adım gradyanı bağımsız formül ve sonlu farkla sınanır; gözlenen veri, sonuç veya eşikler uyum sağlatmak için değiştirilmez.

## Bilimsel sınırlar

Önceki GRM audit'inde olduğu gibi, motorlar arası bilimsel fark kabul ölçütleri henüz bağımsız kalibre edilmiş değildir. Sayısal farklar ölçülür; küçük olmaları otomatik PASS veya “mükemmel uyum” anlamına gelmez. Aynı fixture'da yakınsama, popülasyon geneli kurtarma/SH/GA kapsamı/model uygunluğu ya da global optimum kanıtı değildir. Parametre SH ve EAP posterior SS farklı niceliklerdir; yeni R SH/GA eşdeğerliği iddiası yoktur. Gerçek fit sonucu ölçülmüş fakat sertifikasız olarak kalırsa bu engel PR raporunda açıkça korunur.
