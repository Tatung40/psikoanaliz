# Design B: projected Newton (ön kayıt `7c5b1ab`) — **BOUNDARY SOLUTION**: sayısal olarak durağan KKT noktası, düzenli PASS değil

Tek fit yapıldı. 7 adım kabul edildi; her adım olurlu ve logLik hiç azalmadı. Fit STATIONARY ile durdu. Süre 99.7 s.

| Ölçüt | Sonuç |
|---|---|
| Projected score (GH161) | 2.757 → 3.0e-6; kapı 1e-5, PASS |
| Projected score (GH241) | 3.0e-6; PASS |
| logLik (GH161) | −179462.838063 → −179462.833991 (+0.00407) |
| \|logLik GH241 − GH161\| | 2.3e-10; PASS |
| Aktif yüz bilgisi | PD; en küçük özdeğer 2.67; koşul 2.26e4; PASS |
| Sınır kestirimi (KKT işareti doğru) | I23 c = 0 (skor −1.47); I26 d = 1 (skor +3.12); I27 d = 1 (skor +0.65) |
| R başlangıcından en büyük değişim | .0019 |

Bu üç parametrenin gerçek değerleri iç bölgede: I23 için c ∈ [.10, .20], I26 ve I27 için d ∈ [.80, .90]. Buna rağmen
N = 10000 ve J = 30'da bile ML çözümü bu parametreleri sınıra koyuyor. Sayısal sorun çözüldü. Kalan engel, bu
asimptotların pratik tanımlanabilirliği. Ön kayıt gereği düzenli PASS sıfır sınır kestirimi gerektirdiğinden, geri kazanım
kriteri **uygulanmadı**. Sınırdaki parametreler için Wald çıkarımı düzenli değildir.
Monte Carlo, SE/GA sertifikasyonu ve üretim değişikliği yapılmadı.

## Ön-birleştirme denetimi (yeniden fit yapılmadı)
Kayıtlı parametrelerle yeniden hesaplandı:
- Tüm kestirimler sınırların içinde.
- logLik monoton.
- GH161 ve GH241 projected score 3.0e-6.
- logLik GH161 = GH241 = −179462.833991.
- Üç sınırda da KKT işareti doğru.

`profile_approx.json` dosyasında **ikinci dereceden profil yaklaşımı** var. Bu tam bir yardımcı-parametre profillemesi
değil. GH161 gözlenen bilgisinin Schur tümleyeni kullanıldı. Eğim, sınırdaki skor olarak alındı. Kesim χ²₁/2 = 1.92; sınır
karışımı (χ̄²) göz ardı edildi.

| Parametre | Yaklaşık 95% bölge | Gerçek değerde logLik düşüşü |
|---|---|---|
| I23 c | [0, .35] | 0.37 |
| I26 d | [.66, 1] | 0.87 |
| I27 d | [.68, 1] | 0.32 |

Profil SD .17–.26 aralığında ve üç gerçek değer de bölgenin içinde. Bu, asimptotların zayıf pratik tanımlanmasıyla
tutarlı. Yapısal tanımsızlık ya da global optimallik **iddia edilmez**.

Etiketler:
- BOUNDED KKT STATIONARY: desteklendi.
- REGULAR NUMERICAL PASS: desteklenmedi.
- PARAMETER RECOVERY: sertifikalanmadı.
- SE/CI COVERAGE: sertifikalanmadı.

Düzeltme: `algorithm.maximize` artık olurlu olmayan ya da sonlu olmayan bir başlangıcı sessizce kırpmıyor, hata veriyor.
Bu, kayıtlı fiti değiştirmiyor; başlangıç olurlu ve sonluydu. Sonlu olmayan deneme noktaları zaten reddediliyordu; bu da
test ediliyor.
