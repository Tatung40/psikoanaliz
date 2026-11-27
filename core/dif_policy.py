"""Policy D (hybrid DIF certification) notices. Separate module: core/dif_classical.py, core/dif_center.py and the DIF
views are frozen/protected sources (validation/NON_DIF_IRT_PROTECTED_SOURCES.json); wiring these texts into those
screens requires a declared source amendment."""
OBSERVED_SCORE_NOTICE = (
    "Gözlenen puan DIF (MH/Mantel, lojistik ve sıralı lojistik regresyon): gruplar arasında yetenek farkı (impact) ve "
    "maddelerin ayırt ediciliği farklı olduğunda toplam puan eşleştirmesi yetersiz kalabilir ve I. tip hata şişebilir; "
    "saflaştırma bu eşleştirme yetersizliğini ortadan kaldırmak zorunda değildir. Sayısal doğrulama (difR ile uyum) "
    "çıkarımsal geçerlik kanıtı değildir. p ve düzeltilmiş p değerleri yalnız TARAMA amaçlıdır; I. tip hata/FWER/güç "
    "sertifikası yoktur.")
SCREENING_STATUS = "TARAMA — çıkarımsal olarak sertifikalı değil (sayısal: difR ile uyumlu)"
SIBTEST_NOTE = ("SIBTEST ham toplam puanla değil, çapa puanına gerçek-puan regresyon düzeltmesiyle eşleştirir; Python mirt "
                "sonucu R mirt ile sayısal olarak uyumludur (aynı algoritmanın iki uygulaması, bağımsız algoritma değildir). "
                "p değerleri yalnız TARAMA amaçlıdır.")
