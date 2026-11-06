# G Kuramı — son kanıt matrisi (ön kayıt `f57ac34`)

Kapsam: yalnız `balanced.csv` fikstürü (sha256 `bd385f8c…`); dengeli, rastgele yüzeyli desenler; G çalışması ve iki D senaryosu
(özgün ve ×2 yüzey sayısı). Bu, fikstür düzeyinde bir doğrulamadır; genel sertifika değildir. Ekranda **LIBRARY TESTED** kalır.

Bağımsız referanslar:
- **B:** R 4.3.3 `stats::aov` (`reference.R`; yuvarlama eşlemesi kaldırıldı).
- **C:** gtheoryr 0.2.0 @3b53991 (`gtheoryr_compare.R`).

Yerel ham çıktılar `evidence_final/` dizininde; CI bunları `gtheory-r-diagnostics/` altında yeniden üretiyor.

| Çıktı | p×i | p×r | p×i×r | p×(r:i) | p×(i:r) |
|---|---|---|---|---|---|
| SS/df/MS | V (B, C) | V (B, C) | V (B, C) | V (B) | V (B) |
| Ham varyans bileşenleri | V (B, C) | V (B, C) | V (B, C) | V (B) | V (B) |
| Negatif bileşen / sıfıra kırpma | LT (fikstürde negatif yok) | LT | LT | LT | V (B): `r` ve `p x r` negatif |
| Evren, göreli ve mutlak hata varyansı | V (B, C) | V (B, C) | V (B, C) | V (B) | V (B) |
| Eρ² ve Φ (G ve D ×2) | V (B, C) | V (B, C) | V (B, C) | V (B) | V (B) |
| G/D özdeşliği (özgün sayılar) | V (özdeşlik) | V | V | V | V |
| Göreli/mutlak SEM ve hata bütçesi | V (türetilmiş özdeşlik) | V | V | V | V |
| Afin değişmezlik (y → 100 + 0.001y) | V (gerileme testi) | V | V | V | V |
| gtheoryr | karşılaştırıldı | karşılaştırıldı | karşılaştırıldı | NOT COMPARABLE | NOT COMPARABLE |

V = NUMERICALLY VALIDATED (fikstür kapsamı); LT = LIBRARY TESTED. FAIL: yok.

**En büyük farklar:**
- Bileşen ve hata varyansları: R ile ≤ 3.0e-13, gtheoryr ile ≤ 5.5e-13.
- Katsayılar: ≤ 4.8e-5. Bu, kütüphanenin dört ondalıklı gösteriminden geliyor; sınır 5e-5.

**Düzeltilen kusur:** GeneralizIT 0.1.2 T, bileşen ve katsayıları dört ondalığa yuvarladığı için sonuçlar puan birimine
bağlıydı. ×0.01'de Eρ² .9943 yerine 1.0000 çıkıyor, ×0.001'de analiz çöküyordu. Adaptör artık kütüphaneyi merkezlenmiş ve
ölçeklenmiş puanlarda çalıştırıyor. Motor değiştirilmedi. Regresyon testi: `test_affine_invariance_gd_identity_sem_and_budget`.

**Kenar durumlar (kapalı başarısızlık):** eksik, sonsuz ve metin puan; eksik ya da tekrarlı hücre; sabit puan; tek düzeyli
yüzey; yuvalı kimlik hatası; dengesiz yuvalama; sabit yüzey; desteklenmeyen desen; geçersiz D senaryosu; sıfır birey
varyansı. Eski sonuç ve indirmeler parmak izi değişince siliniyor (UI testi).

**Kapsam dışı ve kalan sınırlar:**
- Sabit yüzeyler; dengesiz veya eksik desenler; (p:r) × i.
- F/p, katsayı güven aralığı, bootstrap ve maliyet optimizasyonu.
- Fikstür sayısı tek. EduG karşılaştırması yok. gtheoryr ile yuvalanmış desen karşılaştırması mümkün değil.
