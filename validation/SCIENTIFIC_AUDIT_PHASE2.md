# Bilimsel doğrulama — Faz 2 (MIRT, G Kuramı, Eksik Veri, MTK grafikleri)

Base: `main` `45d61f7`. PR #136 (çıktı düzeltmeleri) açık ve merge edilmemiş; bu rapor o PR'ın değişikliklerini
içermiyor ve onları tekrarlamıyor. Bu denetimde psikometrik algoritma değiştirilmedi. Doğrulama
yalnız küçük deterministik verilerle yapıldı: iki küçük GRM/GT hesabı, Monte Carlo/bootstrap yok, tam regresyon yok.
Yerel R yok. Pinlenmiş `generalizit==0.1.2` paketi (`requirements.txt`) yalnız denetim için kuruldu;
projeye yeni bağımlılık eklenmedi.

## Doğrulama düzeyleri

| Analiz | Düzey | Kanıt |
|---|---|---|
| G Kuramı p × i (VC, ρ², Φ, D çalışması) | **Doğrulandı (bu fikstür)** | Bağımsız iki yönlü ANOVA / beklenen-MS özdeşlikleri. `validation/generalizability/balanced.csv` puanları r üzerinden ortalandı (p = 6, i = 3). σ²_p 35.22722, σ²_i 6.71722, σ²_pi 0.41889. n_i = 3: ρ² .99605, Φ .93675. n_i = 10: ρ² .99881, Φ .98014. Motorun 4 ondalığa yuvarlanmış değerleri bunlarla birebir eşleşiyor. |
| G Kuramı p × r, p × i × r, yuvalanmış desenler | Kısmen | `validation/generalizability/reference.R` R ANOVA karşılaştırması CI'da çalışıyor; bu denetimde yeniden çalıştırılmadı. |
| Eksik veri: tanı ve atama | **Doğrulandı (değişmezler)** | 40 × 4, %18 eksik, seed 1. Eksik hücre (29) ve eksik satır (22) sayıları doğru. Mode/median/KNN/zero yöntemlerinde gözlenen değerler birebir korunuyor; kimlik sütunu değişmiyor; listwise silme 18 satır bırakıyor; mode atanan değer gerçek mod. İkili maddede median/KNN yuvarlama onayı olmadan engelleniyor. MCAR/MAR/MNAR çıkarımı yapılmıyor (`core/missing_imputation.py:2`). |
| MTK grafikleri: GRM ICC/IIF/TIF/SEM | **Doğrulandı (bağımsız formül)** | N = 600, k = 5, seed 7, tek fit. `coef()` değerlerinden elle yazılan GRM olasılıkları ile `fitted_category_curves` arasındaki fark 1.7e−16. Kategori toplamı = 1 (2.2e−16). Bağımsız Fisher bilgisi Σ(P′)²/P ile IIF arasındaki göreli fark 3.7e−10. IIF ≥ 0 (en küçük .0033). TIF = ΣIIF ve SEM = 1/√TIF farkları 0. |
| MTK grafikleri: 1–4PL eğrileri | Kısmen (statik) | `core/irt_library_curves.py:28–36`: I = (P′)²/[P(1−P)], P ∈ [c, d], parametre kısıtları kontrol ediliyor. Bağımsız sayısal test bu denetimde çalıştırılmadı. |
| Wright haritası (Rasch) | Kısmen (statik) | `core/irt_rasch_wright.py:8–33`: yakınsama, N(0,1) ölçeği ve a = 1 kapıları var; θ (EAP) ile b aynı logit ölçeğinde; histogram toplamı N'ye eşit olmak zorunda. |
| MIRT: parametre, korelasyon, LogLik/AIC/BIC | Kısmen | R mirt Stage 1 sayısal eşdeğerliği CI'da (run 40383677441; `validation/mirt_stage1`). Kodda zorunlu değişmezler var: korelasyon köşegeni = 1 ve pozitif tanımlılık (`core/irt_multidim.py:159`), işaret kanonikleştirmesi kovaryansa doğru yansıtılıyor (`:92`), TIF = ΣIIF assert ile kontrol ediliyor (`:236`), SEM = √diag(I⁻¹) (`:242`). AIC/BIC formülleri doğru (`irt_multidim_general.py:132`, `irt_multidim_factorized.py:79`). Yeniden fit yapılmadı. |
| PCM/GPCM/NRM eğrileri, MIRT yüzeyleri | Doğrulanmamış | Bağımsız sayısal karşılaştırma yapılmadı. |

## Bulgular

| # | Öncelik | Durum | Yer | Beklenen | Gerçek | Etki |
|---|---|---|---|---|---|---|
| 1 | P2 | CONFIRMED | `generalizit` 0.1.2 `design.py` (kütüphane yuvarlaması; `core/generalizability.py` yuvarlama yapmıyor) | Hata varyansları tam hassasiyetteki bileşenlerden hesaplanmalı | Bileşenler önce 4 ondalığa yuvarlanıyor, sonra kullanılıyor. Örnek: göreli hata .139633; tam değer .419/3 = .139630. | Sunum/ondalık hassasiyeti düzeyinde; G/Φ'nin 4. ondalığını etkileyebilir. |

P0: 0 · P1: 0 · P2: 1. Bilinen sertifikasyon sınırları (sabit yüzeyler, dengesiz G desenleri,
tekli atamanın belirsizliği küçük göstermesi, MIRT'te NOT CERTIFIED) yeni hata olarak sayılmadı.

## Ekran → Word grafik tutarlılığı

Plotly şekilleri ekrana verilen nesnenin kendisinden yakalanıyor (`ui/academic_word.py`, `plotly_chart`).
Bu nedenle Word'deki grafik ekrandakiyle aynı veriden çiziliyor; yapısal olarak tutarlı. Piksel düzeyinde
karşılaştırma yapılmadı (NOT VERIFIED).

## Yeniden üretme

Betikler bu raporun hazırlandığı oturumda çalıştırıldı.
- **G Kuramı:** `analyze(x, 'p x i', {'p':'p','i':'i'}, 'score', scenarios={'p':[6],'i':[3,10]})` sonucunu
  MS_p, MS_i, MS_res değerlerinden σ² = (MS − MS_res)/n ile hesaplanan bileşenlerle karşılaştır.
- **GRM:** Veri: `expit(a(θ−b_k))` ile üretilen 600 × 5 matris (seed 7). `fit_b14(df, 'GRM')`;
  olasılık P_k = P*(≥k) − P*(≥k+1); bilgi Σ(dP/dθ)²/P (merkezi fark, h = 1e−6).
- **Eksik veri:** 40 × 4 matris, %20 MCAR maskesi (seed 1); her yöntem için gözlenen hücrelerin eşitliği.

## Doğrulanamayan alanlar

- R karşılaştırmaları yerelde çalıştırılamadı (R yok); mevcut CI kanıtı kullanıldı.
- MIRT 3PL/4PL ve bifaktör sayısal çıktıları.
- PCM/GPCM/NRM ve MIRT yüzey grafikleri.
- Wright haritasında görsel ölçek eşleşmesi.
- KNN atamasının çok değişkenli dağılım özellikleri.

## Öncelikli öneriler

1. G Kuramı hata varyanslarını ve katsayıları tam hassasiyetteki bileşenlerden yeniden hesapla ya da
   yuvarlamanın kütüphaneden geldiğini raporda belirt (P2).
2. PCM/GPCM/NRM eğrileri için GRM'deki gibi bağımsız formül testi ekle.
3. MIRT yüzey SEM'i için analitik 2B fikstür testi ekle (ör. ortogonal 2PL, bilinen yükler).
