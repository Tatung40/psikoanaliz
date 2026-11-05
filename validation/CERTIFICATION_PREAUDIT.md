# Sertifikasyon ön denetimi — main `d9a6266` (PR #136–#139 dahil), 2026-11-05

Bu ön denetimde kod değiştirilmedi ve Monte Carlo çalıştırılmadı. Aşağıdaki her kanıt dosyası açılıp içeriği
okundu; hiçbiri varsayılarak kabul edilmedi. "CERTIFIED" yalnız şu dört koşulun hepsi sağlanan hücrelerde
kullanıldı: bağımsız referans, önceden commit edilmiş kabul ölçütü, arşivlenmiş sayısal sonuç ve dar biçimde
tanımlanmış kapsam. Kanıt eksikliği (GAP) ile bilimsel kusur (DEFECT) ayrı tutuldu.

## 1. İncelenen mevcut kanıtlar

| Kanıt | İçerik (dosyadan okundu) | Ön kayıt sırası | Değerlendirme |
|---|---|---|---|
| `r_mirt_phase1/final_summary.json` | R mirt 1.48 ile en büyük \|Δparam\|: Rasch 1.3e-6, 1PL 4.4e-4, 2PL 9.2e-4, 3PL 2.4e-4 (eşlenmiş kısıtlar); 4PL 0.065, iki motorda da yakınsama yok | — | Rasch/1PL/2PL/3PL nokta eşdeğerliği arşivde. 4PL ML: başarısız (eşdeğerlik yok) |
| `evidence/p6_binary_se/coverage.json` | 1PL/2PL, N = 1000, J = 12, 300 tekrar. Wald kapsaması a .951, b .953; MCSE .004 | `470629c` → `b47b29f` ✔ | Kapsamlı SE/GA kanıtı (yalnız bu koşul) |
| `evidence/p7_binary34_se/coverage.json` | 3PL MAP (N 3000, J 30): a .950, b .967, c .968. 3PL ML: b .827 (N 3000), .887 (N 10000). 4PL ML: b .815, c .884, d .869. 4PL MAP: a .994 (aşırı kapsama) | `2314695` → `b9092f3` ✔ | 3PL MAP GA desteklenir. 3PL/4PL ML ve 4PL MAP desteklenmez; kodda kapalı (`core/irt_binary_inference.py`) |
| `evidence/p4_polytomous/*.json` | GRM/PCM/GPCM, N 500/2000: kapsama .940–.970, ölçüt [.925, .975]. NRM kütüphane SE'si .60–.67 (karşılamıyor); `oakes` uyarlamasıyla .932–.968 (karşılıyor) | belirtilmiş, commit sırası denetlenmedi | SE/GA kanıtı (koşul kapsamında) |
| `evidence/p8_bifactor/recovery.json` | Genel faktör RMSE: N 500 için .494, N 1000 için .271 (ölçüt ≤ .30). Durum: **NOT MET** | — | Bifaktör geri kazanımı N ≤ 500'de yetersiz (bilimsel sınırlılık) |
| `mirt_stage1/phase1_evidence.json` | GRM_3D ve diğer profiller: R ile sayısal eşdeğerlik (run 40383677441) | — | MIRT noktasal eşdeğerliği arşivde |
| `r_mirt_polytomous`, `r_mirt_rsm`, `r_mirt_34`, `generalizability` | Yalnız `compare.py` ve `reference.R` var; CI kapıları (≤ ölçüt) mevcut | — | **Sayısal sonuçlar arşivlenmemiş** → incelenemedi |
| PR #132 / #138 / #139 testleri | STUIRT, POLYEQUATE, Equating Recipes, psych, GPArotation, R mirt (LSAT7); statsmodels, sklearn, pingouin; bağımsız formüller | toleranslar test dosyasında, testten önce | Hesaplama özdeşliği kanıtı |

**PR #139 matrisine iki düzeltme:**
- 3PL SE/GA hücresi NV değil **P** olmalı: 3PL MAP, N 3000 / J 30 koşulunda ölçütü karşılıyor.
- GRM, RSM ve 3PL/4PL için "R ile V" hücreleri **P** olmalı: R sonuçları yalnız CI kapısında, arşivde değil.

## 2. Hücre durumları

| Alan | Durum | Kapsam |
|---|---|---|
| 1PL/2PL Wald SE/GA kapsaması | **CERTIFIED (dar kapsam)** | N 1000, J 12, θ ~ N(0,1), Python mirt varsayılanları (GH21, tol 1e-4). Nokta referansı R mirt (arşivde). Ön kayıt sırası doğrulandı |
| 3PL MAP Wald GA | **CERTIFIED (dar kapsam)** | N 3000, J 30, normal θ, kütüphane önseli. Ön kayıt sırası doğrulandı |
| Hesaplama özdeşlikleri: MH/CMH, Holm/BH/BY/Bonferroni, α/KR-20, λ1–λ6, G Kuramı p×i, EAP, 1–4PL/GRM/PCM/GPCM/NRM olasılık/bilgi/SEM, bağlama/eşitleme, GPA | **CERTIFIED (yalnız hesaplama)** | Formül doğruluğu. Çıkarımsal geçerlilik bu etiketle sertifikalanmaz |
| Rasch/1PL/2PL/3PL nokta kestirimleri | VERIFIED (tek fikstür) | Genel kestirim doğruluğu sertifikalanmadı |
| GRM/PCM/GPCM/NRM SE/GA | VERIFIED (Monte Carlo); commit sırası denetlenmedi | N 500/2000 |
| 4PL ML (kestirim ve çıkarım) | **DEFECT değil, yapısal sınırlılık** | Zayıf tanımlanabilirlik; GA bilinçli olarak kapalı. Sertifikasyon olası değil |
| 3PL/4PL ML Wald GA | Desteklenmiyor (kanıt: eksik kapsama) | Kodda kapalı — doğru davranış |
| Bifaktör, N ≤ 500 | NOT MET (bilimsel sınırlılık) | — |
| 1PL ortak-a SE/GA; MIRT SE/GA/MAP; Lord serbest-SD; MTK-DIF FWER; G Kuramı sabit/dengesiz desenler; NRM madde uyumu | GAP | Kuyruktaki görevler |

## 3. Kritik boşluklar (önceliğe göre)

1. **P0** — R politomöz, RSM, 3PL/4PL ve G Kuramı karşılaştırmalarının sayısal sonuçları arşivde yok. Arşivsiz sonuç sertifikalanamaz. Maliyet düşük; Monte Carlo gerekmez.
2. **P0** — 4PL ML. Sertifikasyon hedeflenmemeli. İstenirse yalnız deterministik bir profil olabilirlik tanısı yapılabilir. Kalıcı NOT SUPPORTED kararı belgelenmeli.
3. **P0** — MIRT/bifaktör çıkarımı kapalı ve MAP doğrulanmamış. Bifaktör geri kazanımı N 500'de yetersiz. UI'da örneklem uyarısının varlığı doğrulanmalı.
4. **P0** — 1PL ortak-a SE/GA: kapsama çalışması yok.
5. **P1** — MTK-DIF (GRM-LR, Lord/Wald) FWER: GRM-LR için yalnız 30 çift var; Lord/Wald için hiç kanıt yok.
6. **P1** — Lord serbest-SD Rasch için R `itemtype = 'Rasch'` karşılaştırması yok.
7. **P1** — G Kuramı p×r, p×i×r ve yuvalanmış desenler: R sonuçları arşivde değil.
8. **P2/P3** — Ekran, Word ve CSV/XLSX tutarlılığı için yalnız seçilmiş rotaların testleri var. Sertifika engeli değil.

## 4. Kaçınılmaz Monte Carlo

Monte Carlo yalnız çıkarımsal olasılık iddiaları için zorunludur, çünkü bunlar deterministik olarak kurulamaz:

- (a) 1PL ortak-a Wald GA kapsaması.
- (b) GRM-LR FWER (sıralı tasarım; #126 planı).
- (c) Lord/Wald 2PL FWER — yalnız sertifika isteniyorsa.
- (d) 3PL MAP GA'nın yeni N/J koşullarına genişletilmesi — yalnız ürün o koşulu iddia edecekse.
- (e) Bifaktör geri kazanımının N ≥ 1000'e taşınması — yalnız iddia edilecekse.

Gerekmeyenler: R nokta eşdeğerliği, formüller, eğriler, dışa aktarımlar ve 4PL ML tanısı (profil olabilirlik deterministiktir).

## 5. Sertifikasyon kabul kriterleri (önceden sabit)

- **Nokta eşdeğerliği (R/Python):**
  - Koşullar: aynı veri hash'i, model, kısıtlar, latent ölçek, kestirici/önsel ve quadrature; ikisi de yakınsamış; yansıtılmış gradyan ≤ 1e-5; sınır çözümü yok.
  - Toleranslar: \|Δparam\| ≤ 1e-3, \|ΔlogLik\| ≤ 1e-3, ICC/TIF göreli fark ≤ 1e-4, SE göreli fark ≤ .02.
  - Sonuçlar JSON olarak, sürüm ve hash'le commit edilir.
- **Kapsama (Wald GA):**
  - Ölçüt: \|kapsama − .95\| ≤ .02; havuzlanmış MCSE ≤ .004 (≥ 3600 aralık); yakınsama ≥ .99; geri tutulan GA ≤ .05; \|ortalama yan\| ≤ .05.
  - Koşullar tek tek değerlendirilir; kapsam yalnız çalışılan N, J ve θ dağılımıdır.
- **FWER:**
  - Ölçüt: Bonferroni ile koşullar arası eşzamanlı, tek yönlü %97,5 CP üst sınırı ≤ .075 (Bradley). "Yakın nominal" etiketi için ≤ .06.
  - Başarısız analizler olay sayılır. Bakış noktaları ve α bölüşümü önceden yazılır.
- **CERTIFIED kapısı:** Yukarıdaki ilgili ölçütler; ön kaydın sonuçlardan önce commit edilmiş olması; arşivlenmiş sonuç; tekrarlanabilir CI çalıştırma bağlantısı; sınır ve kararlılık testleri. Bunlardan biri eksikse NOT CERTIFIED.

## 6. Gözden geçirme

- Kontrol edilen yüksek riskli çıktılar: 4PL/3PL GA, MIRT çıkarımı, NRM SE, DIF p ve FWER, G Kuramı katsayıları, eksik veri ataması.
- Varsayıma dayanan iki nokta, kuyrukta ayrı görev olarak yer alıyor:
  - p4 ön kayıt sırası denetlenmedi (görev Q2).
  - Bifaktör UI uyarısı denetlenmedi (görev Q5).
- Bütün modellerin sertifikalanabileceği iddia edilmiyor. 4PL ML ve N ≤ 500'deki bifaktör, mevcut kanıtla sertifikalanamaz.
