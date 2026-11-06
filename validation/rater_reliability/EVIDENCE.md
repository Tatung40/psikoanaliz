# Puanlayıcı güvenirliği — kanıt matrisi (ön kayıt `e615be4`)

Kapsam: yalnız `examples/rater_reliability` deterministik örnekleri (30 yöntem × 1 örnek; 10 veri seti). Genel sertifika
**değildir**; uygulamada hiçbir yöntem CERTIFIED olarak etiketlenmez. Ham R çıktısı `r_output/reference.csv`,
karşılaştırma `comparison.csv` (154 değer), durumlar `evidence.json`.

**Referanslar:**
- irr 0.85 @98e4535 ve irrCAC 1.4 @720ab22: CRAN GitHub aynalarından kaynak olarak yüklendi.
- psych::ICC ve psych::cohen.kappa.
- R `stats` paketi.
- Python'dan ayrı yazılmış analitik R formülleri.

**En büyük farklar (tolerans içi):**
- Yuvarlanmamış referanslar: 9.2e-14.
- irrCAC: 4.6e-06. irrCAC 5 ondalığa yuvarladığı için tolerans 5e-6.

**Tolerans dışı tek çıktı: ICC(2,k) %95 GA.**
- Referanslar kendi aralarında uyuşmuyor: irr::icc McGraw–Wong (1996) ICC(A,k) Satterthwaite formülünü kullanıyor
  (alt sınır .910753), psych::ICC ise ICC(A,1) sınırlarının Spearman–Brown dönüşümünü (.911490).
- Python sonucu psych ile 1e-13 içinde eşleşiyor, irr ile eşleşmiyor (fark 7.4e-4).
- Ön kayıt gereği GA **gizlendi**. Kestirim, F ve p doğrulandı.

| Yöntem | Doğrulanan çıktılar | Başarısız (gizli) | Referanslar | En büyük fark | Durum |
|---|---|---|---|---|---|
| Cohen Kappa (κ) | ci95, estimate, p, se, z0 | — | irr::kappa2; psych::cohen.kappa | 2.7e-15 | NUMERICALLY VALIDATED |
| Doğrusal ağırlıklı Cohen Kappa (κw (doğrusal)) | ci95, estimate, p, se, z0 | — | irr::kappa2; psych::cohen.kappa | 1.3e-14 | NUMERICALLY VALIDATED |
| Karesel ağırlıklı Cohen Kappa (κw (karesel)) | ci95, estimate, p, se, z0 | — | irr::kappa2; psych::cohen.kappa | 2.2e-14 | NUMERICALLY VALIDATED |
| Fleiss Kappa (κF) | estimate | — | irr::kappam.fleiss; irrCAC::fleiss.kappa.dist | 6.7e-16 | NUMERICALLY VALIDATED |
| Conger Kappa (κC) | estimate | — | irr::kappam.fleiss(exact); irrCAC::conger.kappa.raw | 4.7e-07 | NUMERICALLY VALIDATED |
| Light Kappa (κL) | estimate | — | irr::kappam.light | 5.0e-16 | NUMERICALLY VALIDATED |
| Scott Pi (π) | estimate | — | analytic; irrCAC::fleiss.kappa.raw | 3.1e-06 | NUMERICALLY VALIDATED |
| Gwet AC1 (AC1) | estimate | — | irrCAC::gwet.ac1.raw | 4.5e-06 | NUMERICALLY VALIDATED |
| Gwet AC2 (karesel ağırlık) (AC2) | estimate | — | irrCAC::gwet.ac1.raw(quadratic) | 4.3e-06 | NUMERICALLY VALIDATED |
| Brennan–Prediger katsayısı (BP) | estimate | — | irrCAC::bp.coeff.raw | 3.3e-06 | NUMERICALLY VALIDATED |
| Krippendorff Alfa — nominal (α (nominal)) | estimate | — | irr::kripp.alpha; irrCAC::krippen.alpha.raw | 4.1e-06 | NUMERICALLY VALIDATED |
| Krippendorff Alfa — sıralı (α (sıralı)) | estimate | — | irr::kripp.alpha | 3.3e-16 | NUMERICALLY VALIDATED |
| Krippendorff Alfa — aralık (α (aralık)) | estimate | — | irr::kripp.alpha; irrCAC::krippen.alpha.raw | 4.3e-06 | NUMERICALLY VALIDATED |
| Krippendorff Alfa — oran (α (oran)) | estimate | — | irr::kripp.alpha; irrCAC::krippen.alpha.raw | 4.6e-06 | NUMERICALLY VALIDATED |
| Tam yüzde uyum (Po) | all_agree, estimate | — | irr::agree; irrCAC::pa.coeff.raw | 3.3e-16 | NUMERICALLY VALIDATED |
| PABAK (PABAK) | estimate | — | analytic; irrCAC::bp.coeff.raw | 0.0e+00 | NUMERICALLY VALIDATED |
| Bangdiwala B (B) | estimate | — | analytic | 2.2e-16 | NUMERICALLY VALIDATED |
| ICC(1,1) (ICC1) | F, ci95, df1, df2, estimate, p | — | irr::icc; psych::ICC | 2.5e-14 | NUMERICALLY VALIDATED |
| ICC(1,k) (ICC1k) | F, ci95, df1, df2, estimate, p | — | irr::icc; psych::ICC | 2.5e-14 | NUMERICALLY VALIDATED |
| ICC(2,1) (ICC2) | F, ci95, df1, df2, estimate, p | — | irr::icc; psych::ICC | 9.2e-14 | NUMERICALLY VALIDATED |
| ICC(2,k) (ICC2k) | F, df1, df2, estimate, p | ci95 | irr::icc; psych::ICC | 9.2e-14 | NUMERICALLY VALIDATED (kısmi; başarısız çıktılar gizli) |
| ICC(3,1) (ICC3) | F, ci95, df1, df2, estimate, p | — | irr::icc; psych::ICC | 9.2e-14 | NUMERICALLY VALIDATED |
| ICC(3,k) (ICC3k) | F, ci95, df1, df2, estimate, p | — | irr::icc; psych::ICC | 9.2e-14 | NUMERICALLY VALIDATED |
| Kendall W (W) | chi2, df, estimate, p | — | irr::kendall | 4.4e-20 | NUMERICALLY VALIDATED |
| Spearman sıra korelasyonu (ρs) | estimate, p | — | stats::cor.test | 3.3e-16 | NUMERICALLY VALIDATED |
| Pearson korelasyonu (r) | ci95, estimate, p | — | stats::cor.test | 7.8e-16 | NUMERICALLY VALIDATED |
| McNemar testi (McNemar) | chi2, df, p | — | stats::mcnemar.test | 2.2e-16 | NUMERICALLY VALIDATED |
| Stuart–Maxwell testi (SM) | chi2, df, p | — | analytic | 1.6e-14 | NUMERICALLY VALIDATED |
| Bhapkar testi (Bhapkar) | chi2, df, p | — | analytic | 3.4e-14 | NUMERICALLY VALIDATED |
| Bland–Altman uyum analizi (BA) | bias, ci95_bias, ci95_loa, loa, sd | — | analytic | 4.6e-14 | NUMERICALLY VALIDATED |

**Uygulamada değiştirilen motor davranışları:**
- pingouin 0.6 ICC güven aralıklarını 2 ondalığa yuvarlıyor. Bu yüzden GA'lar aynı McGraw–Wong formülleriyle yuvarlanmadan
  hesaplanıyor ve irr ile psych'e karşı doğrulandı.
- statsmodels `SquareTable` sıfır hücrelere varsayılan olarak 0.5 ekliyor. Bu kapatıldı (`shift_zeros=False`); tekil
  kovaryans durumunda analiz açıklamayla durdurulur.
- Python irrCAC paketi `scipy==1.12` ve `pandas<3` sabitlemesi yüzünden kullanılmadı. Gwet AC1/AC2, Brennan–Prediger ve
  Conger kapalı formla hesaplanıyor; R irrCAC ve irr ile karşılaştırıldı.

**Desteklenmeyen ya da doğrulanmayan çıktılar:**
- Fleiss, Conger, Light, Scott, Gwet, BP, Krippendorff, PABAK ve Bangdiwala için SE/GA/p (Python motoru yok).
- Eksik veride Krippendorff dışındaki yöntemler: analiz açıklamayla durdurulur.
- Dengesiz ICC tasarımları ve bootstrap.

**Değişmezlik ve gidiş-dönüş testleri (|fark| ≤ 1e-12):**
- Puanlayıcı sırası (15 simetrik yöntem).
- Nominal yeniden etiketleme (9 yöntem).
- Geniş ↔ uzun ↔ kategori-sıklık dönüşümü.
- CSV ↔ XLSX gidiş-dönüş (30 örnek).
- Ağırlıklı κ'nın kategori sırasına duyarlılığı.
