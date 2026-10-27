# Genellenebilirlik Kuramı — kapsam ve doğrulama

Motor: **GeneralizIT 0.1.2**, MIT, upstream `v0.1.2` / commit
`8af4faefa4027e5e7e5b45495a5540b2093dc173`.
Kaynak: https://github.com/tylerjsmith111/GeneralizIT .
Bağımlılıkları mevcut NumPy, pandas, SciPy; yeni hesaplama motoru yazılmadı.
Gerçek kurucu `GeneralizIT(data, design_str, response)`; README'deki eski
`input_str` / `levels` örnekleri kullanılmaz. D API: `calculate_d_study(d_study_design=...)`.

| Desen | Ekran / gerçek motor testleri | Bağımsız referans |
|---|---|---|
| p × i | Dengeli, rastgele | İki yönlü ANOVA / beklenen MS kimlikleri + R CI |
| p × r | Dengeli, rastgele | İki yönlü R ANOVA |
| p × i × r | Dengeli, rastgele | Üç yönlü R ANOVA |
| p × (r:i) | Dengeli, rastgele; r kimliği ebeveynler arasında benzersiz | Yuvalanmış R ANOVA |
| p × (i:r) | Dengeli, rastgele; i kimliği ebeveynler arasında benzersiz | Yuvalanmış R ANOVA |
| (p:r) × i ve diğer desenler | NOT VALIDATED / kapalı | Çalıştırılmadı |
| Sabit yüzeyler | Kütüphane API'si var; ekran NOT VALIDATED / kapalı | Bağımsız doğrulanmadı |
| Dengesiz, eksik, tekrarlı veya tek düzeyli veri | Kapalı; sessiz silme/birleştirme yok | Bilimsel destek iddiası yok |

## Yöntem ve sınırlar

Henderson Method I, REML/ML değil. Kütüphane `anova_table` çıktısı
**T (düzeltilmemiş kareler)**, katsayı matrisi ve varyans bileşenlerini içerir;
T hiçbir yerde SS olarak etiketlenmez. Tamamlayıcı SS/df/MS tabloları yalnız
dengeli ortogonal desenlerde ardışık en küçük kareler altuzaylarından elde edilir;
bu hesaplama varyans kestirim motorunu değiştirmez. Rastgele-etki F/p paydaları
ve katsayı güven aralıkları doğrulanmadığı için üretilmez.

Ham negatif varyanslar aynen saklanır. Katsayılarda GeneralizIT'in mevcut
sıfıra kırpması kullanılır; katkı yüzdeleri bu kırpılmış toplam üzerinden
hesaplanır ve ayrı sütun olarak gösterilir. Sıfır birey varyansı çıkarımı engeller.
Kütüphane hem varyansları hem katsayıları dört ondalığa yuvarlar; yeni hassasiyet
uydurulmaz. Hata varyansları, **sürüme sabitlenmiş** `_get_tau_facets`,
`_calculate_tau`, `_get_little_delta_facets`, `_calculate_little_delta`,
`_get_big_delta_facets`, `_calculate_big_delta` kütüphane metotlarıyla okunur;
kamusal API bu değerleri yalnız stdout'a yazar. Sürüm değişirse adaptör durur.

D katsayıları doğrudan kütüphane API'sinden gelir. Pseudo-desen ve hata
varyansları aynı kütüphanenin `create_pseudo_df` ve `_calculate_levels_coeffs`
metotlarını kullanır. Birey sayısı değiştirilemez; yuvalı yüzey düzeyleri
**ebeveyn başına** sayılır. En az ölçme birimi sonucu yalnız girilen adaylar
arasında ve Eρ²/Φ hedeflerinin ikisi için geçerlidir; genel maliyet optimizasyonu
veya gelecekteki katsayının garantisi değildir.

Bulut sınırları: 10000 gözlem, yüzey başına 100 düzey, tamamlayıcı ANOVA'da
500 tasarım sütunu / 2 milyon hücre; en çok 25 D senaryosu ve her biri 10000
pseudo-gözlem. Aşımlar hesaplama başlamadan/ilgili tablo ayrılmadan reddedilir.

## Tekrarlanabilir bağımsız karşılaştırma

`balanced.csv`: 6 birey × 3 madde × 2 puanlayıcı, 36 sabit gözlem.
Puanlar deterministik tam/modüler aritmetik ile tanımlanmış test girdileridir;
örnekleme, Monte Carlo veya fitted referans yoktur. Seed gerekmez. İki yüzeyli
alt örnek ve benzersiz yuvalanmış kimlikler aynı dosyadan türetilir.

Bağımsız referans `reference.R`, **gerçek R stats::aov** kullanır. Dengeli
ortogonal random-effect expected-MS kimlikleriyle bileşenleri hesaplar;
GeneralizIT'i veya Python adaptörünü çağırmaz. ANOVA referansı için REML
`lme4` çıktısını eşdeğer gibi kullanmak yerine aynı ANOVA kestirimi seçilmiştir.
R gtheory/EduG karşılaştırması yapıldığı iddia edilmez.

Önceden belirlenmiş kontroller:

- SS/df/MS: rtol=atol=1e-10 (küçük, iyi koşullu deterministik doğrusal problem).
- Dört ondalığa yuvarlanmış varyanslar: atol=1e-12, rtol=0 (yalnız CSV/binary temsil).
- Hata/evren varyansları: rtol=atol=1e-12; dört ondalıklı katsayılar eşit olmalıdır.
- G tasarımı ve iki kat yüzey düzeyli D tasarımı, her desen için ayrı karşılaştırılır.
- Hata, eksik R veya uyumsuzluk CI'yi düşürür; `REQUIRE_GTHEORY_R=1` skip'i engeller.

Yerel ortamda R yok: **NOT EXECUTED**. Bağımsız testin gerçek durumunu
PR'ın native-R CI işi ve `gtheory-r-diagnostics/` artefaktları belirler;
test çalışmadan başarılı referans sonucu varsayılmaz. Her desenin stdout/stderr,
R sessionInfo, ANOVA/bileşen/katsayı CSV'leri ve fark JSON'u korunur. Başarı etiketi
`NUMERICALLY VALIDATED` yalnız ilgili sabit örneğe aittir; genel sertifika değildir.
Ekran bilinçli olarak **LIBRARY TESTED**, **CERTIFIED değil** durumunu korur.

```bash
Rscript validation/generalizability/reference.R validation/generalizability/balanced.csv 'p x (r:i)' /tmp/g-ref
REQUIRE_GTHEORY_R=1 python -m pytest -q tests/test_generalizability.py::test_independent_r_anova_and_g_d --basetemp=gtheory-r-diagnostics
python -m pytest -q tests/test_generalizability.py tests/test_generalizability_ui.py
```

## UI, dışa aktarma ve oturum güvenliği

Ortak CSV/XLSX yükleme, kenar menüsü callback'i ve Python kod kaydı kullanılır.
Uzun veri veya benzersiz bireyli geniş p×i veri; özgün madde etiketleri korunur.
Verinin gerçek içeriği ve tüm veri/desen/rol/D/hedef ayarlarının parmak izi
değiştiğinde saklanmış sonuçlar ve bütün indirmeler kaldırılır. İndirme callback'i
`ignore`; sonuçlar ve seçili sayfa tekrar çalıştırmada korunur. CSV/XLSX formül
enjeksiyonu koruması ve PNG/ZIP yardımcıları mevcut modülden alınır.
ZIP: ANOVA, ham Henderson, varyanslar, G/hata/D tabloları, PNG'ler, yöntem,
ayarlar ve motor log'u. Ortak Python sekmesi gerçek girdilerle .py ve runtime
ZIP üretir; bağımsız replay testi dosyayı depo dışında çalıştırır.
Kullanıcıya yönelik R kod üretimi eklenmez; mevcut R mirt iş akışları korunur.
