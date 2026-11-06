# 4PL ML kurtarma — sonuç (ön kayıt `dd3d646`)

**Sonuç: iki tasarım da FAIL. Sertifikalı kapsam: NONE. Üretim değişmedi** (`FOUR_PL_ML_CERTIFIED_SCOPE = None`).
Geri kazanım, Monte Carlo ve SE/GA kapsama çalışması ön kayıt gereği **çalıştırılmadı**, çünkü sayısal kapılar geçilmedi.

Ortam: yerel 2 çekirdekli konteyner, R 4.x, mirt 1.48. Ayrıntılar `metadata_A.csv` ve `metadata_B.csv` dosyalarında.
Strateji: önce EM (GH61), ardından mirt'in yerel sınırsız BFGS BL yöntemi GH61 → GH101 → GH161 sırasıyla çalıştırıldı.
Denetim için bağımsız olarak GH241 kullanıldı. Tüm değerlendirmeler ortak değerlendiriciyle yapıldı.

| | A: N3000/J12 fikstürü (değiştirilmedi) | B: N10000/J30, iç ve bilgilendirici parametreler |
|---|---|---|
| Sınıflama | **OUTSIDE COMMON BOUNDS** | **CONVERGENCE FAILURE** |
| GH61 BL | M9: a = 87.7; logLik −23272.321 > Python sınırlı optimumu −23272.575 | Ortak logLik EM başlangıcından **düştü** (−179462.841 → −179462.858), R "converged" bildirdi |
| GH101/GH161 BL | logLik NA; M9 d = .013 < c, M10 b = 26.5 (sayısal çöküş) | Projected score 2.53 / 2.76 |
| GH161 projected score (kapı 1e-5) | tanımsız | 2.76 → FAIL |
| GH241 denetimi | tanımsız | ΔlogLik 5.8e-11; kuadratür sorunu yok |
| Bilgi matrisi | tanımsız (parametreler uzay dışında) | PD, min özdeğer 2.17, koşul 2.8e4 → yerel tanımlama var |
| Sınır kestirimi | — (Python: 11; R bounded EM: M9:a) | 0 |
| Süre | 1 dk | 77 dk (bütçe 180 dk) |

## Tanı
- **A — tanımlanamama.** Sınırsız olabilirliğin üst sınırına a → ∞ yönünde (M9) yaklaşılıyor. Yani ortak kutu içinde iç
  MLE yok. Sınırlı çözüm sınırda kalıyor (Python'da 11 sınır). Bu, kuadratürden ya da optimizerdan bağımsız bir
  zayıf tanımlanabilirlik sorunu. Bu fikstürde sınırlı ML ancak sınırda çözüm verebilir; düzenli Wald çıkarımı yapılamaz.
- **B — optimizer.** Kuadratür kararlı, bilgi iyi koşullu ve çözüm iç bölgede. Buna rağmen R mirt EM ve BL, ortak
  amacın durağan noktasına ulaşmadı. BL, kendi bildirdiği logLik'i düşürürken "converged" döndürdü. Bu da BFGS'nin iç
  amacı/gradyanı (BL.LL/BL.grad) ile customTheta/customPriorFun altında bildirilen olabilirlik arasında tutarsızlık
  olduğuna işaret ediyor.
- **mirt 1.48 hatası.** Sınırlı BL, `stats::optim`'e geçersiz bir yöntem ("nlminb") gönderiyor. Smoke testinde, bu
  satırı L-BFGS-B olarak düzeltmek de ilerleme sağlamadı. Bu yüzden bu yol ön kayda alınmadı.

B'nin GH161 kestirimlerinin gerçek değerlere göre RMSE'si a .31, b .17, c .045, d .059. Bu değerler **yalnız tanı amaçlıdır**:
çözüm durağan olmadığı için geri kazanım kriteri uygulanmadı.

## Kalan engel ve sonraki en küçük deney (ayrıca ön kayıt gerekir)
Python'un sınırlı Newton cilası (PR #145 yöntemi) aynı B verisinde ve aynı kapılarla yeniden çalıştırılmalı. Geçerse,
ön kayıtlı R = 100 Monte Carlo için tahmini maliyet ölçülmeli; bütçe aşılırsa SE/GA sonucu INCONCLUSIVE kalır.
R mirt, 1.48'de BL tutarsızlığı düzelene kadar 4PL ML motoru olarak önerilmez.
