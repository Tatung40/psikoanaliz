# 5000×40 simülasyon verisi: bağımsız MTK ön doğrulaması (2026-10-23)

Kaynak: kullanıcı tarafından sağlanan irt_sim_data.xlsx (data, true_params, missing_map).

- N=5000, k=40, 19429 eksik yanıt (%9.7145).
- Kayıp veri haritası ile NaN konumları birebir eşleşiyor.
- 14 MC4, 17 MC5, 9 open; tüm gözlenen yanıtlar 0/1. Çok kategorili MTK verisi değildir.
- Bağımsız SciPy uygulaması: 31 düğümlü Gauss-Hermite ile N(0,1) theta varsayımı altında MML/EM benzeri alternatif güncelleme; 30 iterasyon üst sınırı; 2PL 22 iterasyonda değişim <0.002.
- İlk deneme 2PL logLik -109868.1172, AIC 219896.2344; 3PL logLik -109748.7858, AIC 219737.5715; 4PL logLik -109678.8676, AIC 219677.7353.
- 3PL ve 4PL için yakınsama, sınır çözümleri ve çoklu başlangıçlar doğrulanmadı. Sonuçlar kesin karşılaştırma değildir.
- Platformun kendi MTK motoru ile sayısal karşılaştırma yapılmadı; bu bir platform PASS raporu değildir.
- Referans parametreler simülasyon üretim parametreleridir; bilinen theta değerleri dosyada yok.
- DIF için grup etiketleri; GRM/PCM için çok kategorili gözlenen yanıtlar gereklidir.
- Çalışma dalı: validation-real-ktk-20261023; ana dala birleştirilmemeli, CI ve referans testleri tamamlanmalı.
