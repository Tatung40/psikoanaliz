# B11 — Üçlü bağımsız doğrulama protokolü

Tarih: 2026-10-23. Her metrik için üç ayrı çıktı zorunlu:

1. **Platform**: PsikoAnaliz/EserPsychometrics `core/` fonksiyonları, sürüm/commit SHA ile.
2. **Harici Python**: uygun bağımsız paketler (örn. scipy, statsmodels, pingouin); paket sürümleri ve kullanılan API ile.
3. **Formül temelli referans**: platform kodundan ve 2. yöntemin aynı tahmin fonksiyonundan bağımsız, açıkça belgelenmiş NumPy/formül hesabı.

Her karşılaştırmada: veri hash'i, N/k, örneklem filtresi, eksik veri stratejisi, model parametrelemesi, yetenek dağılımı, optimizasyon başlangıcı/sınırları, tolerans, üç sayısal sonuç, ikili mutlak/göreli farklar, PASS/WARN/FAIL, açıklama.

- KTK: ortalama, örneklem varyansı, p, madde-kalan korelasyonu, alfa/KR20, SEM, çeldiriciler.
- İkili MTK: 1PL/2PL/3PL/4PL; MML/EM ve yakınsama; parametre eşleştirmesi, ölçekleme, logLik/AIC/BIC, ICC/IIC, theta; yakınsamayan sonuç PASS olamaz.
- Çok kategorili MTK: yalnızca uygun kategorili yanıt verisinde GRM/PCM.
- DIF: MH, uniform/nonuniform lojistik, FDR; grup kodlaması, eşleştirme puanı ve yön konvansiyonu eşleştirilmeli. İstatistiksel DIF doğrudan madde yanlılığı değildir.

**Bilimsel dürüstlük kuralı:** Herhangi bir yol çalıştırılamadıysa 'ÇALIŞTIRILMADI' yazılır; diğer yolların çıktıları onun yerine geçirilmez. GitHub Actions genel başarı durumu, tüm metriklerin üçlü doğrulandığı anlamına gelmez. Test fixture'ları ham kişisel veri içermez.

Gerekli her düzeltme önce izole geliştirme dalında; CI ve referans kıyaslaması sonrası main'e alınabilir.
