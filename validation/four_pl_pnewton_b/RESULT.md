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
