# Analiz kapsam matrisi — fa2e25e (motor kaldırma öncesi) ↔ main

Yöntem: `app.py` rota karşılaştırması, UI'dan geçişli erişilebilir modül kümesi
(`validation/legacy_reachability.py`), aynı makinede golden-master karşılaştırması
(`tools/golden_master.py`, CI "Golden-master equivalence"). Ekran adları değil, çalışan çağrılar
karşılaştırılmıştır.

| Analiz | Önce | Sonra | Eksik / durum |
|---|---|---|---|
| KTK: test, madde, çeldirici, güvenirlik, grup, grafik, rapor, Pingouin | ✓ | ✓ (aynı rota) | yok |
| ABCDE anahtar, 0/1 puanlama, CSV/TSV/TXT/XLSX | ✓ | ✓ | yok (golden 0 fark) |
| Rasch MML sayfası (infit/outfit, WLE, Q3, global uyum, Rasch DIF, Wright) | ✓ | ✓ | yok (yerel kestirici korundu) |
| Python mirt Rasch/1PL–4PL: parametre, ICC/IIC/TIF/SEM, EAP, itemfit, Wright (Rasch) | ✓ | ✓ | yok (golden 0 fark) |
| 1PL–4PL ana sayfa (eğri, TIF, SEM, Excel) | ✓ | ✓ | yok (golden 0 fark) |
| MTK: Karşılaştırma (AIC/BIC) | ✗ (yalnız emekli oturum anahtarı; hiç sonuç yok) | ✓ (Python mirt ML) | giderildi |
| MTK: İkili Rapor (B13) | ✓ (yerel motor) | rota kaldırıldı | çıktılar Python rotasına taşındı (aşağıda) |
| — madde parametresi SH / Wald GA | ✓ | ✓ 1PL/2PL/Rasch; 3PL/4PL sınır parametresinde gerekçe | sınırlılık: 3PL/4PL'de kütüphane kovaryansı sonlu değil |
| — θ dağılımı, kişi–madde haritası, parametre dağılımı, asimptot grafikleri, 300 dpi PNG, Excel/ZIP | ✓ | ✓ (opt-in) | giderildi |
| — S-X², infit/outfit, Q3 | ✓ | ✓ (Python itemfit; Yerel Bağımsızlık ekranı) | yok |
| — motorlar arası karşılaştırma sekmesi | ✓ | ✗ | kasıtlı (motor kaldırıldı) |
| GRM/PCM/GPCM/NRM, B14 rapor | ✓ | ✓ | yok (golden 0 fark) |
| RSM (MML, RSM–PCM LR) | ✓ | ✓ | yok (yerel kestirici korundu) |
| DIF: gözlenen puan, Python mirt çok gruplu, GRM/çok kategorili sabit çapalı, 3PL/4PL bootstrap, çok modelli tarama | ✓ | ✓ | yok |
| MIRT (2PL/GRM/PCM/GPCM/NRM, bloklar, RSM yığını, genel profiller) | ✓ | ✓ | yok |
| Bifaktör örnek verisi | ✗ (varsayılan ayarla analiz edilemiyordu) | ✓ | giderildi (3 özgül grup, yakınsıyor) |
| Boyutluluk/EFA, yerel bağımsızlık/Q3/S-X² | ✓ | ✓ | yok (golden 0 fark) |
| Varsayım denetimi, kayıp veri ve atama | ✓ | ✓ | yok |
| G-Kuramı 5 desen, 15 örnek, D çalışmaları | ✓ | ✓ | yok (golden 0 fark; bağımsız EMS) |
| 117 örnek veri indirmeleri, Python kod üretimi | ✓ | ✓ | yok (3 bifaktör örneği kasıtlı değişti) |
| Asimptotik 3PL/4PL DIF, B10/B11 görünümleri | yönlendirilmemiş (erişilemez) | kaldırıldı | kullanıcı kaybı yok |

## Kalan yerel hesaplamalar — kütüphane kararı (2026-10-29)

| Alan | Karar | Gerekçe ve kanıt |
|---|---|---|
| Rasch MML (`core.rasch`) | B | Python mirt ortak serbest a'lı 1PL aynı modeldir (σ = a, b = a·b′): σ farkı 3e-6, b ≤ 1,7e-5, logLik 2e-7 (`tests/test_native_library_equivalence.py`). Yerel kestirici korunur: kütüphane adaptörü eksik yanıtı (MAR) kabul etmez; WLE, infit/outfit, güvenirlik, Q3 ve Rasch DIF bu kestirime bağlıdır. |
| RSM (`core.irt_rsm`) | B | `fit_mirt` RSM sunmaz; alt düzey `EMEstimator(RatingScaleModel)` σ = 1 altında yerel kestirimle 20/20 veri setinde eşleşir (`validation/RSM_VALIDATION.md`, `tests/test_irt_rsm.py`). Rasch tanımlaması (σ serbest) ve RSM–PCM LR yereldir. |
| Sabit çapalı ikili/GRM/PCM/GPCM DIF | B | Aynı çapa tasarımında genel LR `mirt.fit_multigroup` ile eşit: 2PL/GRM/PCM ≤ 1e-5, GPCM ≤ 5e-4 (tolerans 1e-3). Yerel kestirici korunur: kütüphane yolunda çok gruplu Hessian/SE yok; p/q, Wald GA, tek biçimli/olmayan ayrışımı ve arındırma yalnız yerel yolda doğrulanmıştır. |
| 3PL/4PL bootstrap DIF (`irt_mg34`) | C | Kütüphanenin çok gruplu 3PL/4PL'i c/d'yi gruplarda eşit tutar; yerel tasarım (sınırlı MML, gruplara göre asimptot, parametrik bootstrap) farklı bir modeldir. Sayısal eşdeğerlik iddia edilmez; mevcut boyut çalışması ve testler korunur. |
| Q3 / merkezlenmiş Q3 | A | Yerel uyum ekranında madde parametreleri ve bootstrap yeniden kestirimi zaten Python mirt (`local_fit.primary_refit`); Q3 yalnız artık korelasyonudur. Rasch sayfasındaki Q3 yerel Rasch kestirimine tutarlı bağlıdır. |
| Genel MIRT profilleri / bifaktör | C / A | Serbest asimptotlu çok boyutlu 3PL/4PL `fit_mirt` ile desteklenmez; SciPy prototipi deneysel etiketle korunur. Bifaktör `mirt.bfactor` (A); örnekler varsayılan ayarla yakınsar. N=500/700'de yükler geri bulunur (genel RMSE 0,19/0,12); N=300 öğretim örneğinde bir yük sınıra dayanır, ekran bunu artık açıkça uyarır. |

## Bütünleşik çıktı denetimi — ekran ↔ indirme (PR #94)

Yöntem: her analiz ekranında gösterilen tablolar ile o ekranın CSV/XLSX/ZIP içerikleri karşılaştırıldı;
yeni/değişen bağlantılar gerçek dosya içeriği okunarak test edildi.

| Ekran | Durum öncesi | Düzeltme | Test |
|---|---|---|---|
| Rasch MML | C: güvenirlik, model özeti, global uyum (M2, sınıflama, artık), DIF özeti ve arındırma geçmişi ekranda, indirmede yok | Koşullu Excel + ZIP rapor paketi (yalnız aynı veri/ayar için hesaplanan bölümler) | `test_rasch_b9_streamlit.py` (Excel içeriği oturum sonuçlarıyla birebir) |
| B14 (GRM/PCM/GPCM/NRM) | C: GRM/PCM/GPCM karşılaştırması ekranda, hiçbir indirmede yok; tanılar Excel/ZIP'te yok. P1: tanı/karşılaştırma/ZIP bir önceki kestirime ait olabiliyordu | Karşılaştırma CSV'si; tanı ve karşılaştırma Excel/ZIP'e eklenir; hepsi kestirim belirtecine bağlı | `test_b14_optional_exports.py` |
| Python mirt ikili (SH/GA, 8 grafik, ZIP) | A | — | `test_irt_binary_extras.py` |
| Yerel uyum/Q3/S-X², boyutluluk, RSM, Python mirt DIF, MIRT, G/D, GRM/çok kategorili DIF | A (paketler sonuçtaki tüm tabloları içerir) | — | mevcut testler |

Kabul kapıları: G1 kapsam PASS (silinen çıktı yok) · G2 bütünleşme PASS (yerel yardımcılar aynı kestirime bağlı:
kişi EAP ve Q3 Python mirt eğrileri/parametreleri; Rasch/DIF eşdeğerlik testleri) · G3 sayısal doğruluk PASS
(toleranslar değişmedi) · G4 UI PASS (AppTest) · G5 export PASS (Excel/ZIP içerik okumalı testler) · G6 regresyon:
CI golden-master ve tam regresyon sonucu PR'da raporlanır.

## Bağımsız doğrulama ve parametre geri kazanımı envanteri (PR #95)

| Model | Bağımsız referans | Geri kazanım | Durum |
|---|---|---|---|
| Rasch, 1PL, 2PL, 3PL | R mirt (`validation/r_mirt_phase1`, eşleşme ≤ 9e-4 parametre) | 3PL/4PL 20 tekrar (`p1_3pl4pl`); **1PL/2PL 300 tekrar, SH/GA kapsaması .945–.953 (yeni, `p6_binary_se`)** | doğrulandı (belirtilen koşullarda) |
| 4PL | R mirt yakınsamadı | ML ölçütleri sağlanmadı; MAP N ≥ 3000 | kullanım politikası ile sınırlı |
| GRM, PCM, GPCM, NRM (tek boyut) | **R mirt 1.48 (`validation/r_mirt_polytomous`, CI): 4/4 model ölçütleri sağladı** — parametre ≤ 1,9e-4, olasılık ≤ 3,6e-5, ortak logLik ≤ 2,2e-7, EAP ≤ 3e-5; serbest parametre sayısı eşit. NRM: Python ayna (θ → −θ) yönündeydi (eşdeğer çözüm, hizalandı); R varsayılan tanımlamasıyla yakınsamadı, aynı modelin a1 = 1 tanımlamasıyla yakınsadı | Monte Carlo (`POLYTOMOUS_MC_VALIDATION.md`) | sayısal uyum (N = 300, 8 madde, tek veri seti; genel sertifika değildir) |
| RSM | Python mirt `RatingScaleModel` 20/20 | `rsm_recovery_study.py` | doğrulandı |
| Çok boyutlu 2D–5D, bifaktör | R mirt (`mirt_stage1`, 12 profil) | tekrarlı çalışma yok | geri kazanım NOT VERIFIED |
| Sabit çapalı DIF | yerel ↔ kütüphane LR eşitliği (PR #93) | `dif_validation_study.py` | doğrulandı |
| G-Kuramı | bağımsız EMS + R ANOVA CI işi | — | doğrulandı |
