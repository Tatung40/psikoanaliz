# Design B: bounded Newton refinement (ön kayıt `10b89eb`) — **NUMERICAL FAIL (STAGNATION)**

**Ön kontroller geçti.**
- Eşleme: R logLik ile ortak GH161 logLik arasındaki fark 0.
- Gradyan: analitik ile merkezi fark arasındaki en büyük göreli hata 3.5e-6. Eşik 1e-4.

**Tek fit, adım 0'da durdu.** Newton yönünün beklenen logLik kazancı 0.39 idi. Ancak 10 geri izleme adımının (0.5^0 … 0.5^9)
hepsi ortak kutunun dışına düştü.
- Sebep: iki kestirim, gerçek değerlerinden uzakta ve sınırın hemen yakınında duruyor.
  - I23 için c = 1.7e-4 (gerçek değer ≥ .10).
  - I26 için d = .99992 (gerçek değer ≤ .90).
- Newton yönü bu iki koordinatı sınırın ötesine itiyor.
- Ön kayıtlı adım kuralı yansıtma (projection) içermiyor; olurlu olmayan adımı yalnızca atlıyor. Bu yüzden kabul edilebilecek adım kalmadı.
- Parametreler değişmedi.

| Ölçüt | Sonuç |
|---|---|
| Projected score (GH161) | 2.757 → 2.757; kapı 1e-5, FAIL |
| Projected score (GH241) | 2.757; FAIL |
| \|logLik GH241 − GH161\| | 6e-11; PASS |
| Bilgi matrisi | PD, koşul sayısı 2.8e4; PASS |
| Sınır kestirimi | 0 |
| Geri kazanım | NOT RUN |
| Süre | 15.5 s |

Doğrulanmadı: bu fikstürün yerel maksimumu iç bölgede mi, yoksa sınırda mı.

En küçük gerekçeli sonraki adım (yeni ön kayıt gerekir): aynı başlangıç ve aynı kapılarla tek bir **projected Newton**
(sınırda aktif küme) çalıştırmak. Monte Carlo yok.
