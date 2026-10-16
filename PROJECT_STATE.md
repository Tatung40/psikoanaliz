# PROJECT_STATE.md — PsikoAnaliz Proje Durumu

**Son Güncelleme:** 2026-10-15  
**Mevcut Versiyon:** v1.0.0 (Doğrulanmış ve Çalışır Durumda)  
**Geliştirici:** Doç. Dr. Mehmet Taha Eser  
**Uygulama Durumu:** `http://localhost:8501` üzerinde canlı ve aktif  
**Test Durumu:** 30 / 30 Pytest Testi BAŞARILI (0 Hata)

---

## 1. Mimari Bileşenler Durumu

| Bileşen | Dosya Yolu | Durum | Doğrulama & Notlar |
|---|---|---|---|
| **Veri İşleme** | `core/data_processing.py` | Tamamlandı | CSV, XLSX, SAV okuma, otomatik kodlama/ayırıcı, 4 kayıp veri protokolü |
| **Puanlama Motoru** | `core/scoring.py` | Tamamlandı | Çoktan seçmeli 0/1 puanlama, kuramsal ters kodlama (`Max+Min-X`), Likert aralık denetimi |
| **Test İstatistikleri** | `core/test_statistics.py` | Tamamlandı | Ortalama, varyans, standart sapma ($ddof=1$), $SE_M$, çarpıklık, basıklık, $z/T$ puanları, normlar |
| **Madde Analizi** | `core/item_analysis.py` | Tamamlandı | $p, q, r_{it}$, düzeltilmiş $r_{ir}, r_{pbis}, r_{bis}$ (normal ordinat $y$), Kelley %27 $D$, silindiğinde alfa |
| **Çeldirici Analizi** | `core/distractor_analysis.py` | Tamamlandı | Seçenek frekansları, %27 üst-alt oranları, $D_{opt}$, seçenek-kalan korelasyonu, çalışan/çalışmayan/ters çeldirici taraması |
| **Güvenirlik** | `core/reliability.py` | Tamamlandı | Cronbach Alfa, KR-20 ($\alpha \equiv KR_{20}$ kesin sayısal eşitlik kanıtı), KR-21, İki-yarı (Spearman-Brown, Guttman $\lambda_4$), McDonald Omega, SEM, Feldt GA |
| **Grup Analizleri** | `core/group_analysis.py` | Tamamlandı | Bağımsız $t$-testi, Welch $t$, Levene testi, Cohen's $d$, Tek Yönlü ANOVA, $\eta^2$, KTK madde adillik $\Delta p$ taraması |
| **Girdi Doğrulama** | `core/validation.py` | Tamamlandı | Sabit madde ($s^2=0$) tespiti, tek boyutluluk, küçük örneklem uyarısı, sınır kontrolleri |
| **UI Bileşenleri** | `ui/components.py` | Tamamlandı | Özel CSS, akademik kurumsal başlık, metrik kartları, APA 7 tablo başlık ve notları |
| **Görselleştirme** | `ui/charts.py` | Tamamlandı | 12 interaktif Plotly grafiği (Histogram, Kutu, Q-Q, Güçlük, Ayırt edicilik, Kuadrant, Alfa silindiğinde, Çeldiriciler, Isı haritası) |
| **Raporlama Motoru**| `reports/excel_export.py`, `word_export.py` | Tamamlandı | Çok sayfalı formül korumalı Excel (.xlsx) ve APA 7 akademik Word (.docx) ihracı |
| **Ana Uygulama** | `app.py` | Tamamlandı | 12 sekmeli tam Türkçe Streamlit kullanıcı arayüzü |
| **Test Paketi** | `tests/` | Tamamlandı | 30 / 30 birim ve psikometrik doğrulama testi (2.04 saniyede hatasız geçti) |
| **Örnek Veriler** | `sample_data/` | Tamamlandı | Çoktan seçmeli ham (N=350), İkili (N=300), Likert (N=250), Uç durumlar veri setleri |

---

## 2. Sıfır Bütçe & Yerel Güvenlik Onayı
- 0 TL harcama ile tamamen açık kaynak Python kütüphaneleri (`streamlit`, `pandas`, `numpy`, `scipy`, `statsmodels`, `plotly`, `openpyxl`, `python-docx`, `pytest`) kullanılmıştır.
- Hiçbir ücretli API (OpenAI, Anthropic vb.) çağrılmamaktadır.
- Öğrenci veya katılımcı verileri yerel ortamda işlenir, harici sunucuya iletilmez.

---

## 3. Matematiksel Doğrulama ve Eşdeğerlik Kanıtı
- **KR-20 ve Cronbach Alfa Eşitliği:** İkili verilerde $ddof=1$ örneklem varyansı esasına göre:
  $$\lvert \alpha - KR_{20} \rvert < 10^{-12}$$
  eşitliği `tests/test_reliability.py::test_kr20_and_cronbach_alpha_exact_equivalence` ile doğrulanmıştır.
- **Ters Kodlama Güvenliği:** Ampirik sınırlar yerine kuramsal ölçek sınırları ($Min_{teorik} + Max_{teorik} - X$) zorunlu kılınmış, ampirik min/max sapması engellenmiştir.
