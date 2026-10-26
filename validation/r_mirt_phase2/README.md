# Bağımsız R mirt doğrulaması — Faz 2 / yürütme protokolü

Bu dal, `validation-r-mirt-phase1` içindeki gerçek R `mirt` 1.48 ve Python `mirt` 1.2.0 karşılaştırma altyapısından türetilmiştir. Üretim `main` dalı değiştirilmez.

## Durum ve kapsam
- Rasch, 1PL, 2PL, 3PL: Daha önceki R çalıştırmalarında sayısal yakınlık gözlendi; bilimsel PASS **verilmedi**.
- 4PL: bağımsız R çözümü/sonraki warm-start yakınsama sorunları; **FAIL / NOT CERTIFIED**.
- GRM, PCM, GPCM, NRM, RSM: bağımsız R karşılaştırmaları tamamlanmadı. Bu beş aile önce ayrı ayrı yürütülmelidir.
- Bunlar **10 adlandırılmış aile**dir (Rasch ve 1PL ayrı sayıldığında); Python kütüphanesinin yerel dokuz model raporu Rasch/1PL ayrımı içerir ve RSM harici özel motor gerektirir. RSM Python mirt 1.2.0 tarafından doğrudan desteklenmez.

## Kabul kuralları
1. Aynı, SHA256 kayıtlı ham yanıt matrisi; kategori adları/sırası açık.
2. R `mirt` ve Python `mirt` ayrı çalışan bağımsız kestirimlerdir; yakınsamayan fit PASS sayılmaz.
3. D=1, latent ölçek/kimliklendirme, ML/MAP, kategorik parametrizasyon ve bütün sabit/kısıtlı parametreler eşlenir; eşlenemeyen model `NOT COMPARABLE` kalır.
4. Hem ham parametreler hem de ortak θ ızgarasında kategori olasılıkları, ICC/CRC, madde ve test bilgi fonksiyonları, logLik karşılaştırılır. Ölçek yönü ve etiket permütasyonu, sonuç görülmeden tanımlı kurallarla ele alınır.
5. SE/CI, kişi yeteneği, Wright Map, eksik veri ve sınır yanıtları ayrıca doğrulanır. Hessian/delta dönüşümü doğrulanmadan SE eşitliği iddia edilmez.
6. Kalibre edilmemiş herhangi bir toleransla bilimsel PASS ilan **edilmez**; yalnızca sayısal uyuşma rapor edilir.
7. Artifaktlar, sürüm/seed/parametre/çözücü ayarları, stdout/stderr ve başarısızlıklar saklanır. CI başarı durumu bilimsel onay değildir.

## Yeniden üretim
GitHub Actions üzerinde **Independent R mirt Phase 2 — polytomous execution** iş akışı, `workflow_dispatch` ve model seçimi ile çalıştırılır. Opt-in model seçimi ve minimum koşu yalnızca bilimsel yürütme içindir; otomatik `main` merge yapılmaz.

Önce GRM ve PCM, ardından GPCM/NRM/RSM yürütülür. Karşılaştırmalardan sonra önceden kaydedilmiş başarım sınırları ve çoklu veri kümesi genellemesi ayrıca tasarlanır.
