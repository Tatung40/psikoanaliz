# G-Kuramı genişleme — kapsam ve lisans ön incelemesi

Temel: main `03764ac`, PR #151 dahil. Önkayıt: PROTOCOL.json; sonuç görülmeden kaydedilir.

| Özellik | Mevcut | Hedef / motor | Doğrulama | Durum |
|---|---|---|---|---|
| Beş dengeli rastgele desen | GeneralizIT 0.1.2, ölçek düzeltmesi | Aynen koru | Önceki R aov/gtheoryr arşivleri | LIBRARY TESTED; eski fikstür kanıtları korunur |
| Eksik hücre/dengesiz tek değişken | Kapalı | lme4 REML; aynı beş yapı, tanımlanabilirse | Doğrudan R lmer ve dengeli iç çözümde MoM özdeşliği | Planlandı |
| Kısmi çaprazlama | Kapalı | Aynı varyans çekirdekleri tanımlanabilirse | Çekirdek Gram rank + R tanıları | Planlandı |
| Sabit yüzeyler / yeni genel formüller | Kapalı | Sessiz model dönüşümü yok | Ayrı bilimsel sözleşme gerekir | UNSUPPORTED |
| Çok değişkenli G | Yok | Dengeli p×tek yüzey, kovaryans MoM | R çapraz çarpımlar + analitik özdeşlikler | Planlandı |
| Bileşik G/Phi ve ağırlık | Yok | Matris kuadratik biçimleri | R matris hesabı | Planlandı |
| Ağırlık optimizasyonu | Yok | Genelleştirilmiş özdeğer / sınırlı yerel çözüm | Özdeşlikler ve deterministik kontrol | Planlandı; sınırlı çözüm EXPERIMENTAL |
| PRMSE / VAR | Yok | Tanım doğrulaması şart | facet tek başına bağımsız yöntem kanıtı değildir | NOT CERTIFIED / hesap kapalı |
| Phi-cut | Yok | Kare kayıp modeli, açık kesim | Bağımsız türetim ve R hesabı | Planlandı; sınıflama doğruluğu değildir |
| D tasarım araması | En çok 25 senaryo | Ayrı analitik arama, ≤10000 aday; mevcut sınır korunur | Tam küçük ızgara kontrolü | Planlandı |
| Varyans GA / BLUP tanısı | Yok | lme4 profil, ranef; koşulları açık | Gerçek R yürütmesi | R bekleniyor |

## Sabitlenmiş facet incelemesi

`michaelides/facet@0542448c004719a03510985cdbc21d60f379ce4e`, sürüm 0.0.11, GPL-3.
`gstudy(estimator='lme4')` lmer çağırır; `estimator='aov'` MoM yoludur. Varsayılan auto çok değişkenli formüllerde dışlanan motora yönelir; kullanılmaz.
DESCRIPTION zorunlu `brms` ve `posterior` bağımlılıklarını içerir. Bu nedenle paketin kurulumu/namespace yüklenmesi görevdeki kesin dışlama ile bağdaşmaz. Doğrudan facet karşılaştırması **BLOCKED / NOT EXECUTED**; onun adına sonuç uydurulmaz.
Uygulamaya GPL kaynak kopyalanmayacak. lme4 GPL≥2 ayrı R sürecinden standart API ile çağrılır; R/lme4 yeniden dağıtımı yapılırsa kendi lisans ve kaynak sağlama yükümlülükleri ayrıca incelenmelidir. Depoda genel uygulama lisansı bulunmadı; bundan otomatik hukuki uygunluk sonucu çıkarılmaz.

## Gerçek engeller ve sınırlar

Yerel Rscript yok; yerel R kontrolleri NOT EXECUTED. CI gerçek R çalıştıracak. Uygulama R/lme4 bulunmadığında açık Türkçe hata verir, sessiz başka motora geçmez. GeneralizIT varsayılan dengeli davranışı ve mevcut raporlar korunur. Genel çok değişkenli REML ve keyfi sabit yüzey modelleri bu kanıtla açılmaz.

