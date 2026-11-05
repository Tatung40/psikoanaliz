# 4PL ML — GH101 cilalama ve GH161 denetimi (ön kayıt `daf5e24`, tek çalıştırma)

**Karar: NOT CERTIFIED.** Bilimsel engel doğrulandı. Monte Carlo çalıştırılmadı; üretim kestiricisi değişmedi.

| Kontrol | Sonuç | Kapı |
|---|---|---|
| GH101 Newton cilalama (3 adım; GH61 en iyi profil noktasından, 48 parametre serbest) | projected score 1.5e-6 | ≤1e-5: **PASS** |
| GH161 bağımsız denetim, aynı parametreler | projected score 7.75e-3 | ≤1e-5: **FAIL** |
| logLik, aynı parametreler: GH41 / GH61 / GH101 / GH161 | bkz. `result.json`; \|GH161 − GH101\| = 7.6e-5 | ≤1e-3: PASS |
| Bilgi matrisi, aktif yüz | pozitif tanımlı, koşul ≤ 1e8 | PASS |
| Sınırda kalan koordinat | 11/48 (c = 0, d = 1, a = 5) | — |
| Parametre değişimi, GH61 → GH101 | en fazla 0.0158 | — |

## Kök nedenler (engel tablosu)

| # | Sorun | Kanıt | Sınıf |
|---|---|---|---|
| 1 | Durağanlık kuadratüre duyarlı. Skor GH61'de 0.078, GH101 çözümünü GH161 ile denetleyince 0.0078; azalıyor ama 1e-5 eşiğine yaklaşmıyor | bu çalışma; `FOUR_PL_ML_DECISION.md` | Sayısal (sabit GH ızgarası) |
| 2 | Pratik zayıf tanımlanabilirlik: u .5–1 aralığında 2ΔlogLik = .075, a 1.6–4.9 arasında değişiyor | `FOUR_PL_ML_DECISION.md` | Pratik (yapısal değil) |
| 3 | Sınır çözümleri yaygın | bu çalışma: 11/48; p1: sistematik sınır | Pratik |
| 4 | Geri kazanım ölçütleri hiçbir N ≤ 10000 koşulunda karşılanmadı | `evidence/p1_3pl4pl` | Bilimsel sınırlılık |
| 5 | Wald GA kapsaması başarısız: b .815, c .884, d .869 (ön kayıtlı) | `evidence/p7_binary34_se` | Çıkarım geçersiz |
| 6 | R mirt EM yakınsamadı | `evidence/r_34.json` | Bağımsız referans yok |
| 7 | Model karşılaştırmasına uygunluk | yukarıdaki maddeler | Kapı kanıta bağlandı ve kapalı |

## Yapılan değişiklik

`core/irt_model_comparison.py`: koşulsuz `model == "4PL"` dışlaması kaldırıldı; yerine kanıta bağlı
`four_pl_gate` geldi. Kapsam kaydı `FOUR_PL_ML_CERTIFIED_SCOPE = None` olduğu sürece 4PL tabloda istatistikleriyle
görünür kalır ama seçilmez; dışlama nedeni ve sınır kestirimi sayısı gösterilir. Kapıyı açabilecek tek şey,
ileride yapılacak bir sertifikasyon kaydıdır. Kapı sıfır sınır kestirimi de şart koşar.

## En küçük savunulabilir sonraki deney

1. Sabit GH yerine kişiye özgü uyarlamalı kuadratürle (veya GH201/GH301 dizisiyle) aynı fikstürde durağanlığın
   ızgara inceldikçe yakınsayıp yakınsamadığını gösteren deterministik bir dizi. Ayrı bir bağımsız yoğun denetim içermeli.
2. Ancak 1. geçerse ve yalnız Wald yerine profil olabilirlik GA'sı için: N ≥ 10000, J ≥ 30 koşulunda ön kayıtlı tek
   bir kapsama kampanyası. Ölçüt: \|kapsama − .95\| ≤ .02, MCSE ≤ .004.
