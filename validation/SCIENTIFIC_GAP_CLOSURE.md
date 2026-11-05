# Bilimsel eksiklik kapatma — Faz 3 (2026-11-05)

Base: `main` `45d61f7`. Açık PR'lar (#136 çıktı düzeltmeleri, #137 Faz 2 raporu) ezilmedi.
Hiçbir psikometrik algoritma değiştirilmedi. Değişenler: yeni kanıt testleri ve iki etiket metni.
Toleranslar testten önce sabitlendi (`tests/test_scientific_gap_closure.py`, modül başı açıklaması).
Monte Carlo yapılmadı; testlerde yalnızca tek ve küçük deterministik kalibrasyonlar var.

## Envanter (core/ui/reports/tests, hedefli arama)

| Etiket | Satır | Sınıf |
|---|---|---|
| NOT VERIFIED | 11 | A: 1PL ortak-a, binary politika varsayılanı · C: 4PL ML, Lord eşit ölçek/serbest-SD · D: R yok |
| NOT VALIDATED | 5 | B/C: G Kuramı sabit yüzeyler, dengesiz desenler, (p:r)×i; MIRT MAP |
| NOT CERTIFIED | 83 | Kapsam etiketi; sertifika, önceden kayıtlı Monte Carlo gerektirir (C) |
| Koşullu skip | 5 | D: Rscript ×2, kanıt dosyası henüz yok ×2 · B: Wright haritası yalnız Rasch |
| TODO | 1 | `core/group_analysis.py:139`: yöntemsel uyarı metni, eksik işlev değil |

Toplam ele alınan eksiklik: **12** (A 6, B 2, C 3, D 1). Faz 2'de belirtilen kanıt boşlukları da dahil.

## Sonuç tablosu

| Model/Modül | Önceki durum | Yeni durum | Kanıt | Kalan engel |
|---|---|---|---|---|
| GRM eğrileri / IIF / TIF / SEM | Kısmen (Faz 2'de geçici betik) | **VERIFIED** (tek fikstür, bağımsız formül) | olasılık ≤1e-10; Fisher Σ(P′)²/P, göreli 1e-6; TIF=ΣIIF ve SEM=1/√TIF ≤1e-12 | — |
| PCM eğrileri / bilgi | NOT VERIFIED (Faz 2) | **VERIFIED** | P_k ∝ exp Σ a(θ−δ_v), aynı toleranslar | — |
| GPCM eğrileri / bilgi | NOT VERIFIED (Faz 2) | **VERIFIED** | aynı | — |
| NRM eğrileri / bilgi | NOT VERIFIED (Faz 2) | **VERIFIED** | P_k ∝ exp(s_kθ + c_k), aynı | NRM madde uyumu ve Wright haritası kapalı (C) |
| 1–4PL eğrileri / bilgi | Kısmen (statik) | **VERIFIED** (analitik) | sonlu fark türevi, göreli 1e-6; P ∈ [c, d]; IIF ≥ 0; TIF = ΣIIF | — |
| 1PL ortak serbest a | NOT VERIFIED | **Nokta kestirimleri NUMERICALLY VERIFIED**; SE/GA NOT VERIFIED | Yerel σ-serbest Rasch (`core.rasch`) ile özdeşlik: ΔlogLik 1.4e-8, \|a−σ\| 3.4e-7, \|b−b′/σ\| 7.6e-6, ΔAIC 2.7e-8 | R karşılaştırması ve Monte Carlo yok (D/C) |
| G Kuramı p×i (VC, ρ², Φ, D çalışması) | Kısmen (R CI) | **VERIFIED** (bağımsız ANOVA, kalıcı test) | 5e-5 (kütüphane 4 ondalık raporluyor) | Kütüphane bileşenleri 4 ondalığa yuvarlıyor (P2, Faz 2) |
| G Kuramı sabit yüzeyler / dengesiz desenler | NOT VALIDATED (kapalı) | Değişmedi | — | B/C: bağımsız referans ve geliştirme gerekiyor |
| Wright haritası (1PL ortak-a) | Yalnız Rasch | Değişmedi | Eşdeğerlik ölçek uyumunu destekliyor | B: arayüz geçidi ve ayrı arayüz testi gerekiyor |
| 4PL ML, Lord serbest-SD Rasch, MIRT MAP | NOT VERIFIED / VALIDATED | Değişmedi | — | C/D: R ve önceden kayıtlı Monte Carlo gerekiyor |

**Giderilen eksiklik:** 7 (6 kanıt eksikliği + 1 eski etiket). **Açık kalan:** 5 (4PL ML, Lord serbest-SD,
G Kuramı sabit/dengesiz desenler, MIRT MAP, Wright 1PL) ve R bulunmayan ortam. Hiçbir özellik CERTIFIED ilan edilmedi.

## Etiket değişiklikleri

- `core/irt_python_mirt_report.py`: "NOT VERIFIED — ortak serbest a" → "POINT ESTIMATES NUMERICALLY VERIFIED
  (yerel σ-serbest Rasch eşdeğerliği, tek fikstür); SE/GA, R ve Monte Carlo: NOT VERIFIED …".
- `core/irt_binary_usage_policy.py`: 1PL ortak-a uyarısına aynı sayısal eşdeğerlik bilgisi eklendi.
  Uyarı düzeyi `warning` olarak kaldı.

## Testler

`tests/test_scientific_gap_closure.py` (7 test) seçili CI listesine eklendi. Etkilenen etiket testleri yeniden çalıştırıldı.
Tam regresyon PR'da `<!-- full-regression -->` ile tek sefer CI'da çalıştırılıyor.
