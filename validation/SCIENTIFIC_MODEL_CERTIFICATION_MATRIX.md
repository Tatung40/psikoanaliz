# Bilimsel model × iddia matrisi — 2026-11-05

**Kural: SCIENTIFICALLY VERIFIED OR NOT CERTIFIED.** Bu matriste hiçbir model CERTIFIED değildir.
Sertifikasyon için bağımsız referans, önceden kayıtlı kabul kriteri, sınır/kararlılık testleri, çıkarımsal
iddialar için simülasyon ve tekrarlanabilir CI sonucunun hepsi gerekir. Hiçbir modelde bunların tamamı yok.
Tek fikstürle yapılan karşılaştırma yalnız noktasal eşleşmeyi gösterir.

Base: `main` `45d61f7`. Açık PR'lar: #136 (çıktı biçimi), #137 (Faz 2 raporu), #138 (eğri, 1PL ortak-a ve G Kuramı kanıtı).
PR'a özgü kanıtlar ilgili hücrede belirtildi.

Kısaltmalar: **V** = VERIFIED, **P** = PARTIAL, **NV** = NOT VERIFIED, **NS** = NOT SUPPORTED (bilinçli olarak kapalı), **NA** = NOT APPLICABLE.

| Model | Parametre | Uyum (LL/AIC/BIC) | SE/CI/p | Kişi (EAP) | Eğriler / Wright | Kararlılık | Genel durum |
|---|---|---|---|---|---|---|---|
| Rasch / 1PL (a = 1) | V — R mirt 1.48 ≤4.4e-4 (`r_mirt_phase1`); yerel ↔ R 1.3e-6 | V — AIC/BIC özdeşliği; R logLik | P — Monte Carlo 300/300, b GA kapsaması .953 (yalnız çalışılan koşullar) | V — bağımsız kareleme ≤1e-10 (bu PR) | V eğriler (#138); P Wright (statik) | P — yakınsama kapıları | NOT CERTIFIED |
| 1PL ortak serbest a | P — yerel σ-serbest Rasch ile özdeşlik, tek fikstür (#138) | V — ΔLL 1.4e-8 (#138) | NV | V (ortak EAP fonksiyonu) | V eğriler; NS Wright | P | NOT CERTIFIED |
| 2PL | V — R mirt (`r_mirt_phase1`) | V | P — SE kapsama çalışması (`binary_se_coverage_study`); S-X² yalnız LSAT7 (#132) | V | V | P | NOT CERTIFIED |
| 3PL | P — MAP N ≥ 3000, ML N ≥ 10000; R eşlenmiş fit (`r_mirt_phase1/matched_fit.py`) | P | NV — SE/GA kapsaması doğrulanmadı; bootstrap DIF p kapalı | V | V eğriler (#138) | P — sınır çözümleri raporlanıyor | NOT CERTIFIED |
| 4PL | NV — ML: R yakınsamadı, zayıf tanımlanabilirlik; MAP: P (çalışılan koşullarda nokta geri kazanımı) | NV | NS — GA kapalı | V | V eğriler | NV — ML | NOT CERTIFIED |
| GRM | V — R mirt politomöz (`r_mirt_polytomous`) | V | P — politomöz Monte Carlo (`p4_polytomous`) | V | V (#138) | P | NOT CERTIFIED |
| PCM / GPCM | P — Monte Carlo geri kazanımı (`p4_polytomous`) | V | P | V | V (#138) | P | NOT CERTIFIED |
| RSM | P — 1020 fit (`p3_rsm`), R RSM (`r_mirt_rsm`) | V | P | V | P — statik | P | NOT CERTIFIED |
| NRM | P — Monte Carlo; SE kusuru `oakes` ile düzeltildi | V | P — ağırlıklı kategori etkisi GA'sı kapalı | V | V eğriler; NS madde uyumu ve Wright | P | NOT CERTIFIED |
| MIRT (1–4PL/GRM, bifaktör) | P — R Stage 1 sayısal eşdeğerlik (run 40383677441) | V — formüller | NS — interaktif SE/GA/LR kapalı; MAP NV | P | P — kodda yüzey değişmezleri | P | NOT CERTIFIED |
| DIF: MH, lojistik, düzeltmeler | V — statsmodels / sklearn ≤1e-10 / ≤1e-5 (bu PR) | NA | P — impact altında sıralı DIF FWER şişmesi belgelenmiş (#118–#121) | NA | NA | V | NOT CERTIFIED |
| DIF: Lord/Wald (2PL), GRM-LR | V — R mirt ≤7.4e-4 (#130); GRM GH101/GH161 kapıları | NA | NV — FWER yalnız 30 çift (NOT CERTIFIED); 1PL Lord yalnız betimsel | NA | NA | V — sayısal kapılar | NOT CERTIFIED |
| KTK (α, KR-20/21, λ1–λ6) | V — pingouin α ≤1e-6 (bu PR); psych λ (#132) | NA | P — Feldt GA | NA | NA | V — tanımsız katsayı artık NaN (bu PR) | NOT CERTIFIED |
| G Kuramı | V — p×i bağımsız ANOVA (#138); diğer dengeli desenler P (R CI) | NA | NS — F/p ve GA üretilmiyor | NA | NA | P | NOT CERTIFIED |
| Eksik veri | V — gözlenen değer ve maske bütünlüğü (Faz 2 betiği) | NA | NS — tekli atama belirsizliği yansıtmaz | NA | NA | V | NOT CERTIFIED |

Ekran–Word–CSV/XLSX eşdeğerliği tüm modeller için **P**: p biçimi ve Word bölüm düzeltmeleri PR #136'da (merge edilmedi).

## Bu PR'da yeni doğrulanan iddialar (önceden belirlenen toleranslar)

| İddia | Bağımsız referans | Tolerans (gerekçe) | Sonuç |
|---|---|---|---|
| Holm, BH, BY ve Bonferroni düzeltmeleri; NaN korunuyor | statsmodels 0.15 `multipletests` | 1e-10 (kapalı form) | Geçti |
| MH α, SE(log OR) (RGB) ve süreklilik düzeltmeli CMH χ²/p | statsmodels `StratifiedTable` | 1e-10 | Geçti; ETS Δ işareti doğru |
| İkili lojistik DIF LR (tek biçimli, tek biçimli olmayan, omnibus) | scikit-learn, cezasız LBFGS | 1e-5 (iki farklı optimizer) | Geçti |
| Cronbach α; KR-20 = α | pingouin 0.6 | 1e-6 (proje değerleri 6 ondalığa yuvarlıyor) | Geçti |
| EAP θ ve posterior SD | Aynı ızgarada bağımsız kareleme | 1e-10 | Geçti |

## Doğrulanan ve düzeltilen hata

`core/reliability.py` (α, KR-20, KR-21, standart α, λ4 ve iki Spearman–Brown katsayısı): toplam puan varyansı 0
olduğunda ya da korelasyon tanımsız olduğunda katsayılar **0.0** olarak raporlanıyordu. Bu değerler matematiksel
olarak tanımsızdır ve 0 yanıltıcıdır. Düzeltme: değer artık NaN oluyor; Word'de "Raporlanmadı" görünüyor.
Test: `test_undefined_reliability_is_nan_not_zero`.

## Başarısız karşılaştırmalar

Yok. Testin kendi kurulumundaki iki hata (yanlış sütun indeksi; ETS Δ büyüklüğü için keyfî −1 eşiği) düzeltildi.
Hiçbir tolerans gevşetilmedi. Test yalnız Δ'nın işaretini iddia ediyor; ölçülen Δ = −0.71.

## Açık bilimsel riskler ve kalan NV/NS alanları

4PL ML; 3PL SE/GA; 1PL ortak-a SE/GA; MIRT MAP ve çıkarımsal istatistikler; Lord serbest-SD Rasch;
GRM-LR ve diğer IRT DIF yöntemlerinde FWER (yalnız 30 çift); G Kuramı sabit yüzeyler ve dengesiz desenler;
NRM madde uyumu ve Wright haritası; Wright haritasının 1PL'e genişletilmesi. Toplam: **NV 9, NS 8 hücre**.
Yerelde R yok; R kanıtları CI çalıştırmalarından alındı.

## CI

Testler: `tests/test_independent_references.py` (6 test). Tam regresyon PR'da `<!-- full-regression -->` ile tek sefer çalıştırılıyor.

## Q1–Q3 kanıt eki (2026-11-05; yalnız bu görevlerin kapsamı)

GRM/PCM/GPCM/NRM için `evidence/r_polytomous.json`, RSM için `evidence/r_rsm.json`, 3PL/4PL için `evidence/r_34.json`, mevcut dengeli G Kuramı referansları için `evidence/r_gt.json` asıl R çıktıları ve sayısal karşılaştırmaları korur. Bu alanlar **PARTIAL / NOT CERTIFIED**: eski V işareti genel bilimsel sertifika değildir; native quadrature/eksik stationarity kayıtları kapsamı sınırlar.

p4 ve p8 önkayıt atalıkları Q2'de doğrulandı; p4 NRM yöntem düzeltmesi sonuç sonrası, sayısal eşikler değişmemiştir. N500 p4 kapsaması ve N500 bifaktör geri kazanımı sertifikalanmadı. `q1_q3/preregistration.json`.

4PL ML için `FOUR_PL_ML_DECISION.md`: beş yerel GH61 profil noktası stationarity sağladı, aynı çözümler GH101'de sağlamadı. Zayıf pratik tanımlama/sayısal duyarlılık vardır; bu tek fikstürden kalıcı veya yapısal desteklenemezlik çıkarılmaz. Üretim SE/GA kapıları değişmedi. **Yeni CERTIFIED hücre:0.** Q4–Q10 etiket/uygulama senkronu bu batch kapsamında yapılmadı.
