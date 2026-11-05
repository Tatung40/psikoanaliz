# jMetrik / psychometrics-irt 4PL — sonuç (ön kayıt `f4366dd`)

**Java: FAIL — tamamlanamadı.** Kazanan yok. **BLOCK**: Java adaptörü yazılmadı, üretim motoru değişmedi. Monte Carlo yapılmadı.

| Çalıştırma | Commit | Sonuç |
|---|---|---|
| CI job 127398711773 | `e3ec0b1` | Derleme başarılı (pinlenmiş `1fac2d7`, Java 8, Maven). Sürücü, fikstürdeki `1.0` biçimli değerleri okuyamadı (`Byte.parseByte`). Hata bilimsel değil, girdi biçimiyle ilgili |
| CI job 127400205616 | `059202a` | Değerler aynı, yalnız tamsayı biçimine çevrildi (eşitlik assert ile kontrol edildi). Ön kayıtlı tek ML kestirimi (önselsiz, GH61, D = 1, converge 1e-6, maxIter 2000) **43,9 dakikada bitmedi** (17:08:42 → 17:52:38 UTC); iş 45 dakikalık zaman aşımıyla iptal edildi. Kestirim, logLik veya EM günlüğü üretilmedi |

**Kaynak incelemesinden çıkan bulgular:**
- Ortak parametre uzayıyla yapısal eşdeğerlik yok. M-adımı sınırsız Uncmin kullanıyor; kutu kısıtı yok. Olasılık [0, 1] aralığına kırpılıyor. Durma kuralı yalnız parametre değişimine bakıyor; gradyan kontrolü yok.
- Depodaki tek 4PL kestirim testi (`pl4Test`) devre dışı ve Beta4 önselleri kullanıyor, yani MAP. Önselsiz ML 4PL için yazarın kendi doğrulaması yok.

**Lisans:** Java dosyalarının başlıkları ve `pom.xml` içindeki `<licenses>` Apache-2.0. Kök `pom.xml` başlık yorumundaki GPL-3 notu eski bir kalıntı gibi görünüyor; bu çelişki çözülmeden ikili dağıtım önerilmez.

**Karşılaştırma (önceki kanıtlar):**
- Python mirt: GH101'de durağan, GH161 denetiminde FAIL (PR #145).
- R mirt: durağan değil, skor ≈ 4.75 (PR #146).
- Java: çalıştırma tamamlanmadı. Üç motorun hiçbiri kapıları geçmedi; ön kayda göre geri kazanım deneyi başlatılmadı.

**Sonraki en küçük deney (ayrıca ön kayıt gerektirir):** Aynı ayarlarla, EM günlüğü her döngüde akan ve maxIter süre
bütçesine sığacak biçimde önceden sabitlenen tek bir çalıştırma. Amaç yavaşlığı ıraksamadan ayırmak. Bu bir ayar araması değildir.
