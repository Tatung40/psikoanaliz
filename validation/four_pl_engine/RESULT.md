# 4PL motor seçimi — sonuç (ön kayıt `0602389`; CI run 42065126768, job 127377020331, commit `83037a3`)

**WINNER: NONE — karar D: NO VALIDATED 4PL ENGINE.** Üretim varsayılanı değişmedi (KEEP).
Sertifikasyon engellendi (BLOCK). Monte Carlo yapılmadı.

| Aday | Kendi ızgarasında projected score | GH161 denetimi | logLik GH161, Python'a göre | Sonuç |
|---|---|---|---|---|
| Python mirt (PR #145 GH101 çözümü, yeniden kullanıldı) | 1.5e-6 | 7.75e-3 | — | **FAIL** (GH161 denetimi) |
| R mirt 1.48, EM, GH61 (yeni fit) | 4.758 | 4.755 | −0.00411 | **FAIL** (durağan değil) |
| R mirt 1.48, EM, GH101 (yeni fit) | 4.746 | 4.746 | −0.00452 | **FAIL** (durağan değil) |
| R TAM / ltm | — | — | — | **UNSUPPORTED**: kurulu API'de (`tam.mml.3pl`, `tpm`) tahmin edilen üst asimptot yok |

**R ile Python farkları (ortak değerlendirici, θ ∈ [−4, 4]):** en büyük \|Δparam\| .051; \|ΔICC\| .0064–.0080; \|ΔIIF\| .013.
R çözümünün logLik değeri Python'un durağan noktasından .004 daha düşük. Yani R EM, TOL = 1e-8 ve NCYCLES = 20000
altında optimuma ulaşmadan durdu; bu, arşivdeki R yakınsama sorunuyla tutarlı.
Sınır kestirimleri: Python 11, R GH61 ve GH101 birer (M9:a). Eksiksiz liste `ci_summary_run42065126768.txt` dosyasında.

**Kanıtın kökeni:** Artifact blob deposuna bu ortamdan erişilemedi (403). Bu nedenle sonuçlar, aynı çalıştırmanın
CI günlüğündeki `compare.py` özetinden alındı (`ci_summary_run42065126768.txt`). Tam `engine_report.json`,
artifact 'four-pl-engine-3841dbd…' içinde duruyor.

**Kalan engel:** Hiçbir motor durağanlık ve denetim kapılarını birlikte geçemedi. Sonraki en küçük deney:
kişiye özgü uyarlamalı kuadratür veya R mirt'te `method='QMCEM'`/`SEM` yerine EM sonrası sınırlı Newton cilalama.
Ayrıca ön kayıt yapılmalı ve ortak denetim aynı kalmalı.
