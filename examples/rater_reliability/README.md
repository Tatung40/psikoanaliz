# Puanlayıcı güvenirliği örnek verileri

`build.py` deterministik olarak üretir (tohum 20261106; kişisel veri yok). `registry.json`, 30 yöntemin her biri için şunları
tutar: veri dosyası, SHA256, analiz yapılandırması, sütun açıklamaları, ölçme düzeyi, puanlayıcı sayısı, veri düzeni, eksik
veri kuralı, senaryo ve bağımsız R referansından gelen beklenen değerler. Tüm veri setleri `rater_reliability_examples.xlsx`
dosyasında ayrı sayfalar olarak da bulunur. `Aciklama` sayfası sütunların anlamını verir.

**Veri biçimleri:**
- Geniş (varsayılan): satır = puanlanan birim, sütun = puanlayıcı.
- Uzun: `long_format_example.csv`.
- Kategori-sıklık: `fleiss_kappa_counts.csv`.

Şablonlar: `wide_template.csv`, `long_template.csv`, `counts_template.csv`.

`invalid_teaching.csv`, hata tespitini göstermek için bilerek hatalı hazırlandı: tekrarlanan kimlik, eksik puan ve metin
olarak kodlanmış sayı içerir.
