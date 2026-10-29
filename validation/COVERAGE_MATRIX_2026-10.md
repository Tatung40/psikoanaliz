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
