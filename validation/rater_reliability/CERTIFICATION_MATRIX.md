# Puanlayıcı güvenirliği — yöntem × çıktı matrisi

`evidence.json` kaydından üretildi (`matrix.py`), yeniden hesaplama yapılmadı. Doğrulama yalnız örnek veriler için geçerli;
**CERTIFIED: 0**. Tüm çıktıları doğrulanmış yöntem: 29/30. Kısmi: 1.
NOT CERTIFIED (genel kapsam): 30. Kanıtsız ya da uyuşmayan çıktı ekranda ve dışa aktarımlarda gösterilmez.

| Yöntem | Çıktı | Durum | Referanslar |
|---|---|---|---|
| Cohen Kappa | estimate | NUMERICALLY VALIDATED | irr::kappa2; psych::cohen.kappa |
| Cohen Kappa | se | NUMERICALLY VALIDATED | irr::kappa2; psych::cohen.kappa |
| Cohen Kappa | ci95 | NUMERICALLY VALIDATED | irr::kappa2; psych::cohen.kappa |
| Cohen Kappa | z0 | NUMERICALLY VALIDATED | irr::kappa2; psych::cohen.kappa |
| Cohen Kappa | p | NUMERICALLY VALIDATED | irr::kappa2; psych::cohen.kappa |
| Doğrusal ağırlıklı Cohen Kappa | estimate | NUMERICALLY VALIDATED | irr::kappa2; psych::cohen.kappa |
| Doğrusal ağırlıklı Cohen Kappa | se | NUMERICALLY VALIDATED | irr::kappa2; psych::cohen.kappa |
| Doğrusal ağırlıklı Cohen Kappa | ci95 | NUMERICALLY VALIDATED | irr::kappa2; psych::cohen.kappa |
| Doğrusal ağırlıklı Cohen Kappa | z0 | NUMERICALLY VALIDATED | irr::kappa2; psych::cohen.kappa |
| Doğrusal ağırlıklı Cohen Kappa | p | NUMERICALLY VALIDATED | irr::kappa2; psych::cohen.kappa |
| Karesel ağırlıklı Cohen Kappa | estimate | NUMERICALLY VALIDATED | irr::kappa2; psych::cohen.kappa |
| Karesel ağırlıklı Cohen Kappa | se | NUMERICALLY VALIDATED | irr::kappa2; psych::cohen.kappa |
| Karesel ağırlıklı Cohen Kappa | ci95 | NUMERICALLY VALIDATED | irr::kappa2; psych::cohen.kappa |
| Karesel ağırlıklı Cohen Kappa | z0 | NUMERICALLY VALIDATED | irr::kappa2; psych::cohen.kappa |
| Karesel ağırlıklı Cohen Kappa | p | NUMERICALLY VALIDATED | irr::kappa2; psych::cohen.kappa |
| Fleiss Kappa | estimate | NUMERICALLY VALIDATED | irr::kappam.fleiss; irrCAC::fleiss.kappa.dist |
| Conger Kappa | estimate | NUMERICALLY VALIDATED | irr::kappam.fleiss(exact); irrCAC::conger.kappa.raw |
| Light Kappa | estimate | NUMERICALLY VALIDATED | irr::kappam.light |
| Scott Pi | estimate | NUMERICALLY VALIDATED | analytic; irrCAC::fleiss.kappa.raw |
| Gwet AC1 | estimate | NUMERICALLY VALIDATED | irrCAC::gwet.ac1.raw |
| Gwet AC2 (karesel ağırlık) | estimate | NUMERICALLY VALIDATED | irrCAC::gwet.ac1.raw(quadratic) |
| Brennan–Prediger katsayısı | estimate | NUMERICALLY VALIDATED | irrCAC::bp.coeff.raw |
| Krippendorff Alfa — nominal | estimate | NUMERICALLY VALIDATED | irr::kripp.alpha; irrCAC::krippen.alpha.raw |
| Krippendorff Alfa — sıralı | estimate | NUMERICALLY VALIDATED | irr::kripp.alpha |
| Krippendorff Alfa — aralık | estimate | NUMERICALLY VALIDATED | irr::kripp.alpha; irrCAC::krippen.alpha.raw |
| Krippendorff Alfa — oran | estimate | NUMERICALLY VALIDATED | irr::kripp.alpha; irrCAC::krippen.alpha.raw |
| Tam yüzde uyum | estimate | NUMERICALLY VALIDATED | irr::agree; irrCAC::pa.coeff.raw |
| Tam yüzde uyum | all_agree | NUMERICALLY VALIDATED | irr::agree; irrCAC::pa.coeff.raw |
| PABAK | estimate | NUMERICALLY VALIDATED | analytic; irrCAC::bp.coeff.raw |
| Bangdiwala B | estimate | NUMERICALLY VALIDATED | analytic |
| ICC(1,1) | estimate | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(1,1) | F | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(1,1) | df1 | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(1,1) | df2 | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(1,1) | p | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(1,1) | ci95 | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(1,k) | estimate | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(1,k) | F | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(1,k) | df1 | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(1,k) | df2 | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(1,k) | p | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(1,k) | ci95 | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(2,1) | estimate | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(2,1) | F | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(2,1) | df1 | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(2,1) | df2 | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(2,1) | p | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(2,1) | ci95 | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(2,k) | estimate | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(2,k) | F | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(2,k) | df1 | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(2,k) | df2 | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(2,k) | p | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(2,k) | ci95 | WITHHELD (referans uyuşmazlığı) | irr::icc; psych::ICC |
| ICC(3,1) | estimate | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(3,1) | F | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(3,1) | df1 | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(3,1) | df2 | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(3,1) | p | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(3,1) | ci95 | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(3,k) | estimate | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(3,k) | F | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(3,k) | df1 | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(3,k) | df2 | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(3,k) | p | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| ICC(3,k) | ci95 | NUMERICALLY VALIDATED | irr::icc; psych::ICC |
| Kendall W | estimate | NUMERICALLY VALIDATED | irr::kendall |
| Kendall W | chi2 | NUMERICALLY VALIDATED | irr::kendall |
| Kendall W | df | NUMERICALLY VALIDATED | irr::kendall |
| Kendall W | p | NUMERICALLY VALIDATED | irr::kendall |
| Spearman sıra korelasyonu | estimate | NUMERICALLY VALIDATED | stats::cor.test |
| Spearman sıra korelasyonu | p | NUMERICALLY VALIDATED | stats::cor.test |
| Pearson korelasyonu | estimate | NUMERICALLY VALIDATED | stats::cor.test |
| Pearson korelasyonu | p | NUMERICALLY VALIDATED | stats::cor.test |
| Pearson korelasyonu | ci95 | NUMERICALLY VALIDATED | stats::cor.test |
| McNemar testi | chi2 | NUMERICALLY VALIDATED | stats::mcnemar.test |
| McNemar testi | df | NUMERICALLY VALIDATED | stats::mcnemar.test |
| McNemar testi | p | NUMERICALLY VALIDATED | stats::mcnemar.test |
| Stuart–Maxwell testi | chi2 | NUMERICALLY VALIDATED | analytic |
| Stuart–Maxwell testi | df | NUMERICALLY VALIDATED | analytic |
| Stuart–Maxwell testi | p | NUMERICALLY VALIDATED | analytic |
| Bhapkar testi | chi2 | NUMERICALLY VALIDATED | analytic |
| Bhapkar testi | df | NUMERICALLY VALIDATED | analytic |
| Bhapkar testi | p | NUMERICALLY VALIDATED | analytic |
| Bland–Altman uyum analizi | bias | NUMERICALLY VALIDATED | analytic |
| Bland–Altman uyum analizi | sd | NUMERICALLY VALIDATED | analytic |
| Bland–Altman uyum analizi | loa | NUMERICALLY VALIDATED | analytic |
| Bland–Altman uyum analizi | ci95_bias | NUMERICALLY VALIDATED | analytic |
| Bland–Altman uyum analizi | ci95_loa | NUMERICALLY VALIDATED | analytic |
