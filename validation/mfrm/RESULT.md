# Çok yüzeyli Rasch ölçme (MFRM) — doğrulama sonucu

**Ön kayıt:** `b3b9e42`. **Değişiklik (amendment):** `b9ca733`. mfrmr'nin gizil dağılımı TAM ile eşlendi; toleranslar
değişmedi. Bu ön kayıttan önce yalnız bir TAM araç çalıştırması ve log-olabilirlik eşleme kimliği yapılmıştı.

**Model:** Model (marginal ML, theta ~ N(mu, sigma^2)):. Kestirim: R TAM 4.3-25 `tam.mml.mfr`, marjinal ML (EM), 61 düğüm,
conv 1e-7, Ramsay hızlandırması. Tanımlama: madde, yüzey ve eşik parametreleri sıfıra toplanır; μ ve σ² kestirilir.

**Bağımsız referanslar:**
- **mfrmr 0.2.4** `fit_mfrm`: RSM, MML, `population_formula = ~ 1`, 61 Gauss–Hermite noktası. Amaç fonksiyonu aynı;
  kareleme ve optimizer farklı.
- **rasch 1.12.1** `rasch_mfrm`: ikili koşullu ML. Amaç fonksiyonu farklı olduğu için **yalnız gerçek değerin geri kazanımı**
  karşılaştırıldı.
- **Python kimlik kodu:** log-olabilirlik, EAP, WLE ve gözlenen bilgi.
- Ham çıktılar: `reference.csv`, `comparison.csv`, `tam_output/`.

| Çıktı | Durum | En büyük fark | Dayanak |
|---|---|---|---|
| Madde, puanlayıcı ve ek yüzey ölçüleri (logit) | CERTIFIED | 1.7e-06 | mfrmr MML within preregistered tolerance + Python identity |
| Yüzey ölçülerinin standart hatası | PARTIAL | 3.0e-08 | formula/identity evidence or agreement without a preregistered tolerance; definition documented |
| Yüzey ölçülerinin %95 Wald aralığı | PARTIAL | — | formula/identity evidence or agreement without a preregistered tolerance; definition documented |
| Derecelendirme ölçeği eşikleri | CERTIFIED | 9.0e-07 | mfrmr MML within preregistered tolerance + Python identity |
| Eşik standart hataları | PARTIAL | — | formula/identity evidence or agreement without a preregistered tolerance; definition documented |
| Birey WLE ölçüsü ve SE | PARTIAL | — | formula/identity evidence or agreement without a preregistered tolerance; definition documented |
| Birey EAP ve sonsal SS | PARTIAL | 5.4e-06 | PARTIAL: TAM-reported EAP differs from the Python posterior at the final parameters by up to 1.8e-06 (> preregistered 1e-6 identity tolerance); the ap |
| Infit/Outfit MnSq (EAP tabanlı artık) | PARTIAL | 5.5e-07 | formula/identity evidence or agreement without a preregistered tolerance; definition documented |
| Standartlaştırılmış uyum (ZSTD) | NOT CERTIFIED | — | no independent evidence; withheld |
| Gözlenen ve beklenen ortalama puan | PARTIAL | 4.4e-07 | formula/identity evidence or agreement without a preregistered tolerance; definition documented |
| Ayırma, tabaka ve güvenirlik | PARTIAL | 1.4e-05 | formula/identity evidence or agreement without a preregistered tolerance; definition documented |
| Log-olabilirlik, sapma, AIC, BIC | CERTIFIED | 4.2e-06 | mfrmr MML within preregistered tolerance + Python identity |
| Gizil dağılım ortalaması ve varyansı | CERTIFIED | 5.4e-07 | mfrmr MML within preregistered tolerance + Python identity |
| Madde × puanlayıcı etkileşimi LR testi | NOT CERTIFIED | — | no comparison could be made (e.g. TAM did not converge); withheld |
| Etkileşim (yanlılık) kestirimleri | NOT CERTIFIED | — | no independent evidence; withheld |
| Etkileşim kestirimi SE | NOT CERTIFIED | — | no independent evidence; withheld |
| Adil ortalama (diğer yüzeyler ortalamada) | PARTIAL | — | formula/identity evidence or agreement without a preregistered tolerance; definition documented |
| Kategori sıklıkları ve ortalama ölçüler | PARTIAL | — | formula/identity evidence or agreement without a preregistered tolerance; definition documented |
| Puanlayıcı katılık sıralaması | CERTIFIED | 1.7e-06 | ranking of certified facet measures |
| 4+ yüzeyli model | NOT CERTIFIED | — | TAM, önceden kaydedilen ayarlarla dört yüzeyli fikstürde sayısal olarak başarısız oldu (`accel` NaN); uygulama reddediyor |

**Kontroller:**
- Kimlik (Python–TAM): log-olabilirlik ≤ 7.3e-7, WLE ≤ 1.8e-6.
- **EAP kimliği 2/3 fikstürde ön kayıtlı 1e-6 sınırını aştı** (en çok 2.0e-6). TAM'ın EAP'si son M adımından önceki sonsal
  dağılımdan hesaplanıyor. Uygulama, son parametrelerle yeniden hesaplanmış Python EAP'sini gösteriyor; durumu **PARTIAL**.
- Geri kazanım (dengeli ve dengesiz fikstür): TAM ≤ 0.082, rasch PCML ≤ 0.089 logit. Ölçüt: max(3 SE, 0.15).
- TAM `se.xsi` kullanılmadı: kovaryansı yok sayan bir yaklaşım veriyor (K1: 0.032; gözlenen bilgi ile 0.042).
  Raporlanan SE'ler Python gözlenen bilgisinden geliyor ve mfrmr ModelSE ile 3e-8 içinde uyumlu.
- Etkileşim fikstüründe TAM'ın madde × puanlayıcı modeli 3000 iterasyonda yakınsamadı; mfrmr yakınsadı. LR testi
  **raporlanmadı** ve tahmin ayarları değiştirilmedi.

**FACETS tarzı işlevler:**

| İşlev | Durum |
|---|---|
| Ortak logit ölçeği; madde, puanlayıcı ve eşik ölçüleri; puanlayıcı katılık sırası; Wright haritası; kategori eğrileri | Var (CERTIFIED / PARTIAL) |
| Infit/Outfit MnSq | Var. **EAP tabanlı artık**; FACETS JMLE tanımıyla aynı değil |
| ZSTD | Yok (NOT CERTIFIED) |
| Adil ortalama | Var. Tanım: diğer yüzeyler sıfırda; FACETS tanımıyla karşılaştırılmadı |
| Ayırma, tabaka ve güvenirlik | Var (PARTIAL) |
| Yanlılık/etkileşim tabloları, ek yüzeyler (görev, oturum), çapalama, JMLE | Yok ya da gizli |
| FACETS yazılım çıktısıyla kıyas | Yapılmadı: yasal olarak erişilebilir FACETS çıktısı yok. Eşdeğerlik iddia edilmez |

**Veri ve gizlilik kuralları:**
- Bağlantısızlık, yüzey başına birey–düzey grafiğiyle denetleniyor. Bağlantısız tasarım, tekrarlanan satır, boş ara
  kategori, tek düzeyli yüzey ve tam sayı olmayan puan analizi açıklamayla durduruyor.
- Eksik puanlar sayılıp analiz dışı bırakılıyor.
- Birey kimlikleri varsayılan olarak takma adla gösteriliyor.

Uygulama R TAM gerektiriyor; yoksa analiz Türkçe açıklamayla reddediliyor ve başka kestirici kullanılmıyor.
