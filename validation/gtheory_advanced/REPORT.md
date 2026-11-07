# G-Kuramı sıkçı genişleme — bilimsel sonuç

Önkayıt **29dd0c2bede3e9cd29114537321bffdb9843aed5**, temel main **03764ac** (PR #151 dahil).
Mevcut GeneralizIT motoru, beş desenin ANOVA sonuçları, ölçek düzeltmesi ve önceki aov/gtheoryr arşivleri değişmedi.

## Gerçek R yürütmesi ve sınırlı karar

[Run 42323674735](https://github.com/Tatung40/psikoanaliz/actions/runs/42323674735), başlık commit **f52b3f3**,
gerçek yürütme merge commit **fa509747e5ce65565df02eb10dc74413803fdd35**; ham artifact **13554700692**.
R **4.6.1**, lme4 **2.1-0** (R gösterimi 2.1.0), Matrix **1.7.5**, jsonlite **2.0.0**.
CRAN mirror kaynak incelemesi @4370052; fiilen kurulan paket RSPM'nin 2.1-0 sürümüdür. Bu, ikili dosyanın GitHub
commit'inden tarafımızca derlendiği iddiası değildir. facet kaynak incelemesi @0542448, sürüm 0.0.11; kurulmadı.

| Desen / yöntem | Gerçek örnekler | Sayısal bulgu | Bilimsel durum |
|---|---:|---|---|
| REML p×i | Dengeli + 3 eksik hücre | Yakınsamış, tekil değil; aynı R lmer bileşenleri ve G/D/SEM | NUMERICALLY VALIDATED, bu örnekler |
| REML p×r | Dengeli + 3 eksik hücre | Aynı kontroller başarılı | NUMERICALLY VALIDATED, bu örnekler |
| REML p×i×r | Dengeli + 3 eksik hücre | Aynı kontroller başarılı | NUMERICALLY VALIDATED, bu örnekler |
| REML p×(r:i) | Dengeli + 3 eksik hücre | Aynı kontroller başarılı | NUMERICALLY VALIDATED, bu örnekler |
| REML p×(i:r) | Dengeli + 3 eksik hücre | İki yolda da p:r=0, tekillik, geçersiz Hessian; G/Phi/D kapalı | **NOT CERTIFIED**; güvenli ret doğrulandı |
| Matris MoM p×i, 2 boyut | Aynı 32 olay × 2 boyut | R aov polarizasyonuyla kovaryanslar eşleşti | NUMERICALLY VALIDATED, kayıtlı örnek |
| Boyut/bileşik G/Phi, SEM, D | İki ağırlık, 4/8 yüzey | Bağımsız R kuadratik biçimleri eşleşti | NUMERICALLY VALIDATED, bu örnekler |
| Phi-cut | Kesim 0/40, aynı ağırlık/D ailesi | R kare kayıp oranıyla eşleşti | NUMERICALLY VALIDATED hesap; sınıflama/GA sertifikası değil |
| Profil varyans GA | p×i, 3 bileşen | R profil uçları, doğru SD² dönüşümü | Native R hesap testi başarılı; **kapsama sertifikası yok** |
| Ağırlık optimizasyonu | Genelleştirilmiş özdeğer / SLSQP | Amaç fonksiyonu özdeşliği ve sınırlar test edildi | LIBRARY TESTED; SLSQP **EXPERIMENTAL / LOCAL ONLY** |
| Maliyet/Pareto D araması | Sınırlı deterministik ızgara | Aday maliyeti/hedef/nondominasyon kontrolü | LIBRARY TESTED / analitik; arama dışı optimalite yok |
| Sabit yüzey, genel MV REML, özel evren/hata kümeleri | Çalıştırılmadı | Ayrı model/kanıt gerektirir | UNSUPPORTED |
| PRMSE / VAR | Çalıştırılmadı | Tahmin hedefi bağımsız yöntemle netleştirilmedi | NOT CERTIFIED / kapalı |
| facet gstudy aov/lme4 | Çalıştırılmadı | Zorunlu brms/posterior görevde dışlanmış | BLOCKED / NOT EXECUTED |

**Sayısal farklar:** Adaptör–doğrudan R lmer varyans farkı en çok **0.0** (aynı kestirici ve aynı standartlaştırılmış girdi;
bu, farklı iki kestiricinin bağımsız yakınsaması değildir). Dört dengeli iç çözümde bağımsız ANOVA/MoM karşılaştırmaları
önkayıtlı `rtol=2e-4, atol=2e-4×ölçek` sınırları içinde geçti. Dengesiz/sınır örnekleri ANOVA ile eşdeğer sayılmadı.
Çok değişkenli R–Python mutlak farkları: evren **3.55e−15**, yüzey **9.77e−15**, artık **4.86e−15**;
önkayıtlı ölçekli 1e−10 sınırı değişmedi.

Profil varyans GA'ları: p **[9.777249, 76.205312]**, i **[0.692193, 20.897308]**, p:i **[0.923878, 3.138593]**.
Bunlar parametre profil aralıklarıdır; SEM aralığı değildir. Normal rastgele-etki modeline koşulludur;
ampirik %95 kapsama veya genel sağlamlık gösterilmedi. Monte Carlo sayısı **0**.

## Başarısızlıklar nasıl korundu?

İlk run 42323358733 kalibrasyona ulaşmadı: `packageVersion` 2.1-0'ı 2.1.0 yazdığı için metin karşılaştırması durdu.
Paket sürümü değiştirilmeden sayısal sürüm karşılaştırması düzeltildi.

İkinci run gerçekte **10 test geçti, 2 test başarısızdı**. İki başarısız kontrol, her veri setinin yakınsamasını zorunlu
sayan test varsayımıydı. Aynı kaydedilmiş iki R yolu da aynı p:r=0 sınırını ve tekilliği gösteriyor; uygulama zaten
katsayıları NA bırakmıştı. Test sözleşmesi, önkayıtlı güvenli ret politikasına düzeltildi: **bağımsız R'nin reddiyle
aynı ret + boş G/Phi/D zorunlu**. Hiçbir uyum başarılıya çevrilmedi; p×(i:r) için G/Phi **NOT CERTIFIED** kaldı.

Yeniden kestirim yapılmadı. `evidence.json` 181 kanonik ham metin dosyasını, başarısız çözüm çıktılarını, veri/model/ayarları,
orijinal test kaynağını, original CI failure durumunu ve ZIP SHA-256'sını korur. Yalnız pytest `current` takma adları
(aynı dosyaların kopyaları) elendi. Sonraki denetim, aynı ham R/Python bileşenlerinden farkları ve güvenli reddi yeniden hesaplar.
Güncel denetim kaynağı ile gerçek yürütme kaynağı ayrı hash'lenir; eski/yeni veri üreten `fixture` AST'si aynıdır.
Kestirim kodu, veri, tolerans, optimizer ve kapılar değişmedi. Yerel R yok: yeni yerel R yürütmesi iddia edilmiyor.

## Tanım ve genelleme sınırları

- REML, ANOVA MoM veya ML değildir. Sabit kesişim nedeniyle ölçek geri dönüşümünde REML logLik **N−1** artık kontrastını kullanır.
- Dengesiz modelde G/D, **yeni dengeli hedef** katsayısıdır; eşit olmayan gözlenen birey ortalamaları için yanlış güvenirlik sunulmaz.
- Rastgele-etki Gaussian modeli ve yok sayılabilir gözlem seçimi varsayılır; MNAR, dağılım yanlışlığı veya ampirik dayanıklılık sertifikası yoktur.
- Çok değişkenli yol kovaryans bileşenleri kestirir; ayrı tek değişkenli uyumları tam MV model diye birleştirmez. PSD olmayan ham bileşenler kırpılmaz.
- 3–5 boyut API sınırı bağımsız R kanıtının kapsamını büyütmez; fiili R örneği iki boyutludur. Genel/yuvalı/dengesiz MV analiz desteklenmez.
- PRMSE hedefi önemlidir: U=I, hata=I, w=(.5,.5) iken gözlenen C'den T1 için doğrusal projeksiyon PRMSE'si
  `Cov(T1,C)^2/[Var(T1)Var(C)] = .25` olur. Paydada ayrıca Rel(C) kullanmak .50 verir ve farklı hedefe karşılık gelir.
  facet dokümantasyonundaki böyle bir formül bağımsız yöntem doğrulaması sayılmadı; gözlenen/gerçek bileşik ayrımı
  yayımlanmış yöntemle eşlenmeden PRMSE/VAR açılmadı. Bu, facet'in tüm uygulaması hakkında doğrulanmamış bir hata iddiası değildir.
- Kaynaklar: Brennan (2001); Shavelson & Webb (1991); Haberman (2008). Lisans ve zorunlu bağımlılık sınırı GAP_AND_LICENSE.md'dedir.
- R/lme4 canlı sunucuda kurulu varsayılmaz. Eksik runtime açıkça engellenir; uygulama otomatik paket kurmaz veya alternatif kestirici uydurmaz.

## Yazılım doğrulaması ve teslim

Hedefli gerçek GeneralizIT, matris, rank, sınır/PSD, R yokluğu, UI, eski sonuçların bastırılması, Word/CSV/XLSX/PNG/ZIP,
Python offline kaynak paketleme ve geçmiş kanıt bütünlüğü kontrolleri yapıldı. R köprü betiği yalnız onu gerektiren
Python tarifinin offline paketine eklenir; diğer analizlerin Python paketlerine R dosyası konmaz.
Tam regresyon ve son CI sonucu PR'da kaydedilir; tamamlanmadan yeşil veya merge-ready iddiası verilmez.
Merge/deploy yapılmadı. Hiçbir Bayesian bağımlılık veya yöntem eklenmedi.
