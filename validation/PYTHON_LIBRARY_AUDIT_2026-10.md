# Python Kütüphane Analizleri — Bağımsız Denetim (2026-10-29)

Kapsam: yalnız Python kütüphanesi çağrı yolları (mirt 1.2.0, GeneralizIT 0.1.2, pingouin, statsmodels,
scikit-learn). Eski özel motor (`core.irt`, `core.rasch`, `core.irt_mixed`, …) incelenmedi/değiştirilmedi.
Bağımsız referanslar bu denetim için yazılmış numpy/scipy kodudur; R çalıştırılmadı (R doğrulaması mevcut
CI işlerine aittir). Sonuçlar CERTIFIED değildir.

## Kapsam matrisi

| Analiz | Kütüphane / gerçek çağrı | Bağımsız denetim | Sonuç |
|---|---|---|---|
| 1PL/2PL/3PL MML | `mirt.fit_mirt` (`core/irt_python_mirt_adapter.py`) | Kendi GH-61 MML (9 örnek) | ΔlogL ≤ 0.001, max\|Δa,b\| ≤ 0.004 |
| 4PL ML | `mirt.fit_mirt` | Aynı kutu sınırlarıyla MML | Sınırlı optimuma ulaşıyor (ΔlogL ≤ 0.09); a=5 sınırında 5–6 madde → mevcut "not_supported" politikası doğru |
| GRM/PCM/GPCM | `mirt.fit_mirt` | Kendi GH-61 MML (9 örnek) | ΔlogL ≤ 0.18 (21 düğüm etkisi), max\|Δ\| ≤ 0.037 |
| SE (oakes) | `fit.standard_errors` | Sayısal gözlenen bilgi | ≤ 0.0006 fark |
| EAP / posterior SD | `mirt.fscores` | Kendi GH-61 EAP | Birebir (< 1e-5) |
| Bilgi / TIF | `model.information` | Σ(dP/dθ)²/P | Birebir |
| Python mirt kişi tablosu | `core/irt_person_abilities.py` (yerel ızgara EAP, ±4) | `mirt.fscores` | Ortalama ≤ 0.001; uç θ'da ≤ 0.03 (ızgara kesmesi) |
| G/D (5 desen, 15 örnek) | `generalizit` 0.1.2 | Kendi EMS ANOVA, 2×2 D senaryosu | Varyans 4 ondalık eşit; Eρ²/Φ ≤ 5e-5 |
| EFA/KMO/Bartlett | Yerel numpy/scipy | factor_analyzer (yalnız doğrulama) | KMO/Bartlett eşit; yükler ≤ 0.009 |
| ABCDE puanlama 4/20/50 | `core/scoring.py`, `core/answer_keys.py` | Elle eşleştirme + CSV/TSV/TXT/XLSX gidiş-dönüş | Birebir |
| Python mirt DIF | `mirt.fit_multigroup` | Kod denetimi | p/q kapalı, LR betimsel — uygun |

## Giderilen hata

Python mirt 1.2.0 `coef_with_se()` sabit/tanımlama kısıtlı parametreler (PCM a=1, NRM ilk kategori
eğim/kesişim) için SE = 0.0 döndürüyordu; B14 ekranı, CSV'si ve raporu bunu kesin kestirim gibi
gösteriyordu. `coef_with_se_table` bu değerleri boş (NaN) gösterir. Test: `tests/test_python_mirt_fixed_se.py`.

## Eski motor bağımlılıkları — 2026-10-29 kaldırma sonrası durum

- Yerel ikili MTK motoru (`core.irt`, `core.irt_mixed`, B10–B13 altyapısı, motor kaydı) kaldırıldı;
  `tests/test_legacy_reachability.py` hiçbir dosyanın bunları yeniden içe aktarmamasını denetler.
- `core.irt_itemfit` kendi 0/1 doğrulayıcısını taşır; Wright şekli `ui/wright_map.py` içindedir.
- `core.irt_person_abilities` yalnız kütüphane eğrilerini kullanan yerel ızgara EAP'tir (motor değil);
  çıktıları golden-master ile birebir korunur.
- Eşdeğerlik: `tools/golden_master.py` kaldırmadan önce alınan tüm Python çıktılarını (9 model, B14, DIF,
  MIRT, yerel uyum/EFA, 15 G/D, KTK, 117 örnek indirme) kaldırmadan sonra bit düzeyinde yeniden üretir.
- Kütüphane karşılığı olmadığı için korunan yerel kestiriciler: Rasch MML sayfası (`core.rasch`), RSM
  (`core.irt_rsm`), sabit çapalı ikili/GRM/çok kategorili DIF (`irt_mg`, `grm_dif`, `polytomous_dif`),
  3PL/4PL bootstrap DIF (`irt_mg34`), Q3 (`irt_q3`, yerel uyum ekranı da kullanır), genel MIRT profilleri.
