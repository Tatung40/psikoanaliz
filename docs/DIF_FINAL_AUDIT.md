# DIF son denetimi — yöntem × çıktı (2026-11-07, `audit/dif-final`)

Tarihsel belgeler, kaydedildikleri commit'ler için kanıttır. Bu tablo `main` 03764ac ve bu daldaki düzeltmelerin durumunu
gösterir. **Kısaltmalar:**
- **NV:** NUMERICALLY VALIDATED (fikstür kapsamı).
- **LT:** LIBRARY TESTED.
- **EXP:** EXPERIMENTAL.
- **NC:** NOT CERTIFIED.
- **US:** UNSUPPORTED (gizli).

| Yöntem | Uygulama | Doğrulanan çıktılar | Bu dalda giderilen kusur | Bilimsel sınırlılık | Gerekli eylem |
|---|---|---|---|---|---|
| Mantel–Haenszel (ikili) | `dif_classical` | χ², α_MH, ln OR GA, ETS D: NV (statsmodels; **R mantelhaen.test** ≤ 2.8e-13) | Başarısız saflaştırmada p gösteriliyordu | Seçilmiş çapalarda p koşullu (not gösteriliyor); etki altında madde başına hata %4–7 (p2) | — |
| Mantel sıralı ve SMD | `dif_classical` | χ²: NV (genelleştirilmiş CMH; **R analitik** ≤ 1.7e-14); SMD: NV | SMD paydası, referans grubu olmayan katmanları da sayıyordu | Etki altında sıralı FWER %9–12 (arşiv kanıtı; uyarı korunuyor) | — |
| İkili lojistik DIF (tek biçimli / tek biçimli olmayan / omnibus), Nagelkerke ΔR² | `dif_classical` | LR, p, ΔR²: NV (sklearn; **R glm** ≤ 9e-13) | Belirgin negatif LR 0'a kırpılıyordu (p = 1) | Jodoin–Gierl eşikleri sezgisel | — |
| Sıralı lojistik DIF | `dif_classical` (statsmodels OrderedModel) | LR, p, ΔR²: NV (**R MASS::polr** ≤ 7.6e-11) | Kırpma; BFGS hassasiyeti (Newton cilasıyla giderildi); hata sözlüğünde eksik anahtar | Etki altında FWER şişmesi | — |
| Çoklu karşılaştırma (BH, BY, Holm, Bonferroni) | `adjust_pvalues` | NV (statsmodels ≤ 1e-10) | Çok modelli taramada karar düzeltmesiz p < .05 ile veriliyordu; artık BH | Aileler ayrı; ortak FDR iddiası yok | — |
| Rasch / 1PL / 2PL (LR, Wald, Lord, sabit çapa, saflaştırma) | `rasch_dif`, `irt_dif`, `dif_lord` | LR ve parametreler: NV (R mirt 1.48, B9/B10); Lord/Wald: NV (#130) | — | FDR yalnız tek tasarımda (100 tekrar); 30 çift FWER: NC | Daha geniş FWER çalışması (ön kayıt gerekir) |
| 3PL | `irt_dif34` | LR: NV (B11); bootstrap p: **US** (iç-null büyüklük sorunu) | — | Sınır ki-karesi geçersiz | Kapı korunuyor |
| 4PL | `irt_dif34` (`inferential=False`) | Yalnız betimsel | — | Zayıf tanımlanabilirlik; ML durağanlığı sınırda (PR #148–#150) | p kapalı; sertifika beklemede |
| GRM / PCM / GPCM / RSM / NRM | `grm_dif`, `polytomous_dif` | EXP (Python mirt ve analitik referanslar) | — | Genel FDR/kapsam yok; NRM ağırlıklı etki GA'sı **US** (%88 kapsama) | R çok gruplu referansı |
| Çok modelli gözlenen puan taraması | `irt_dif_multimodel_screen` | LT; "MTK DIF DEĞİL" etiketi | BH düzeltmesi eklendi; NRM'de seyrek hücre uyarısı | Permütasyon çözünürlüğü 1/200 | — |
| Etki büyüklüğü, görsel, rapor (CSV/XLSX/DOCX/ZIP/PNG) | DIF görünümleri | Gizlenen çıktılar dışa aktarımda boş ya da "Raporlanmadı"; eski sonuçlar siliniyor (mevcut testler) | Başarısız saflaştırmada "DIF işaretlenen madde" metriği 0 gösteriyordu | Olasılık ölçeğinde GA yok | — |

**Bağımsız karşılaştırma:**
- 256 değer: 12 ikili madde × 14 çıktı ve 8 sıralı madde × 11 çıktı (`validation/dif_classical_r/`).
- Ön kayıt `a44dca6`'da, karşılaştırmalardan önce commit edildi.
- Yalnız optimizer hassasiyeti iyileştirildi: R `polr` reltol 1e-15; Python'da Newton cilası. Toleranslar değişmedi.

**Arşiv kanıtı:** `core/dif_classical.py` değişikliği `validation/SOURCE_AMENDMENTS.json` dosyasında belgelendi.
- Sıralı FWER çalışmaları yeniden çalıştırılmadı.
- Arşiv kayıtlarından etki değerlendirildi: kritik değere 1e-4 yakınlıkta LR yok; en fazla 6/2500 oran değişimi; sonuçlar korunuyor.
- Arşiv metadata'sı değiştirilmedi.
