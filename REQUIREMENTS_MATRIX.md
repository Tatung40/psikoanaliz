# REQUIREMENTS_MATRIX.md — Gereksinimler ve Doğrulama Matrisi

**Proje:** PsikoAnaliz — Klasik Test Kuramı Analiz Platformu  
**Yazar:** Doç. Dr. Mehmet Taha Eser  
**GitHub Deposu:** [https://github.com/Tatung40/psikoanaliz](https://github.com/Tatung40/psikoanaliz) (PRIVATE)  
**Tarih:** 2026-10-16  
**Test Özeti:** 31 / 31 Pytest Testi BAŞARILI (PASSED in 3.65s)

---

| Modül | Gereksinim Kodu | Tanım / İstatistik | Durum | Doğrulama Kanıtı | İlgili Test Dosyası |
|---|---|---|---|---|---|
| **A** | A.1 | CSV, XLSX, SPSS SAV Dosya İçe Aktarma | Tamamlandı | Başarılı dosya okuma ve testler | `tests/test_scoring.py` |
| **A** | A.2 | Otomatik Kodlama (Encoding) ve Ayırıcı Tespiti | Tamamlandı | UTF-8, Windows-1254, virgül/noktalı virgül | `tests/test_scoring.py` |
| **A** | A.3 | Katılımcı ID, Madde Seçimi ve Grup Değişkeni | Tamamlandı | Dinamik sütun haritalama | `tests/test_scoring.py` |
| **A** | A.4 | Çoktan Seçmeli Cevap Anahtarı ile 0/1 Puanlama | Tamamlandı | Harf/boşluk duyarsız eşleştirme | `tests/test_scoring.py` |
| **A** | A.5 | Önceden Puanlanmış (0/1) İkili Veri Desteği | Tamamlandı | Doğrudan analiz entegrasyonu | `tests/test_scoring.py` |
| **A** | A.6 | Çok Kategorili (Likert/Rubrik) Puanlama | Tamamlandı | Sınır kontrolleri ve normalize güçlük | `tests/test_scoring.py` |
| **A** | A.7 | Teorik Sınırlara Dayalı Ters Kodlama (`Max+Min-X`) | Tamamlandı | Kuramsal formül garantisi | `tests/test_scoring.py` |
| **A** | A.8 | Geçersiz Değer ve Mükerrer Katılımcı Tespiti | Tamamlandı | `check_duplicate_participants` | `core/data_processing.py` |
| **A** | A.9 | Kullanıcı Tanımlı Kayıp Veri Kuralları (4 Kural) | Tamamlandı | listwise, zero, mean, keep | `tests/test_missing_data.py` |
| **B** | B.1 | Madde Ortalama, Medyan, Varyans, Standart Sapma | Tamamlandı | $ddof=1$ örneklem varyansı | `tests/test_items.py` |
| **B** | B.2 | Madde Güçlüğü ($p$) ve $q = 1 - p$ | Tamamlandı | Deterministik doğrulama | `tests/test_items.py` |
| **B** | B.3 | Normalize Çok Kategorili Madde Güçlüğü | Tamamlandı | `(Mean - Min) / (Max - Min)` | `core/item_analysis.py` |
| **B** | B.4 | Madde-Toplam Korelasyonu ($r_{it}$) | Tamamlandı | Pearson $r$ korelasyonu | `tests/test_items.py` |
| **B** | B.5 | Düzeltilmiş Madde-Toplam Korelasyonu ($r_{ir}$) | Tamamlandı | Madde-kalan korelasyonu | `tests/test_items.py` |
| **B** | B.6 | Nokta-Çift Serili Korelasyon ($r_{pbis}$) | Tamamlandı | İkili Pearson bağıntısı | `tests/test_items.py` |
| **B** | B.7 | Çift Serili Korelasyon ($r_{bis}$) | Tamamlandı | Normal ordinat $y$ formülü | `tests/test_items.py` |
| **B** | B.8 | Üst-Alt %27 Ayırt Edicilik ($D$) ve Bağ Yönetimi | Tamamlandı | Kelley %27 yüzdelik kesme | `tests/test_items.py` |
| **B** | B.9 | Madde Silindiğinde Alfa Değeri ($\alpha_{-i}$) | Tamamlandı | $k-1$ matris döngüsü | `tests/test_items.py` |
| **B** | B.10| Madde Düzeyinde Pedagojik Tanısal Bayraklar | Tamamlandı | Çok zor/kolay, negatif, zayıf | `tests/test_items.py` |
| **C** | C.1 | Seçenek Frekansları ve Yüzdeleri | Tamamlandı | Sayım ve yüzde hesaplama | `tests/test_distractors.py` |
| **C** | C.2 | Üst-Alt %27 Seçenek Seçim Oranları | Tamamlandı | Çapraz tablo oranları | `tests/test_distractors.py` |
| **C** | C.3 | Seçenek Düzeyinde Ayırt Edicilik ($D_{opt}$) | Tamamlandı | $\%_{üst} - \%_{alt}$ | `tests/test_distractors.py` |
| **C** | C.4 | Seçenek-Kalan Nokta Çift Serili Korelasyon | Tamamlandı | 0/1 seçenek vektör korelasyonu | `tests/test_distractors.py` |
| **C** | C.5 | Çalışmayan ve Ters Çalışan Çeldirici Taraması | Tamamlandı | $< \%5$ ve $D > 0$ tanıları | `tests/test_distractors.py` |
| **D** | D.1 | $N, k$, Ortalama, Medyan, Mod, Ranj, Min, Max | Tamamlandı | Betimsel test istatistikleri | `tests/test_items.py` |
| **D** | D.2 | Varyans ($s^2$), Standart Sapma ($s$), $SE_M$ | Tamamlandı | $ddof=1$ ve $s/\sqrt{N}$ | `tests/test_items.py` |
| **D** | D.3 | Çarpıklık (Skewness) ve Basıklık (Kurtosis) | Tamamlandı | Fisher-Pearson yansız momentler | `tests/test_items.py` |
| **D** | D.4 | Çeyreklikler, Yüzdelikler ve IQR | Tamamlandı | $Q_1, Q_2, Q_3, P_{10}, P_{90}$ | `tests/test_items.py` |
| **D** | D.5 | Standartlaştırılmış $z$, $T$ Puanları ve Yüzdelik Sıra| Tamamlandı | $z=(X-M)/s$, $T=50+10z$ | `tests/test_items.py` |
| **D** | D.6 | Taban ve Tavan Etkisi Yüzdeleri | Tamamlandı | Min/Max frekans oranları | `tests/test_items.py` |
| **E** | E.1 | Cronbach Alfa ($\alpha$) ve Standartlaştırılmış Alfa | Tamamlandı | Varyans ve korelasyon temelli | `tests/test_reliability.py` |
| **E** | E.2 | KR-20 Katsayısı (Alfa ile Özdeşlik Doğrulamalı) | Tamamlandı | $\lvert \alpha - KR_{20} \rvert < 10^{-12}$ | `tests/test_reliability.py` |
| **E** | E.3 | KR-21 Katsayısı (İkili Testler) | Tamamlandı | Eşit güçlük varsayımı formülü | `tests/test_reliability.py` |
| **E** | E.4 | İki-Yarı (Tek-Çift / 1-2 Yarı) & Spearman-Brown | Tamamlandı | Prophecy düzeltmesi | `tests/test_reliability.py` |
| **E** | E.5 | Guttman $\lambda_4$ İki Yarı Güvenirlik | Tamamlandı | Yarı-test varyans formülü | `tests/test_reliability.py` |
| **E** | E.6 | McDonald Omega ($\omega_t$) Kestirimi | Tamamlandı | Tek faktörlü model yükleri | `tests/test_reliability.py` |
| **E** | E.7 | Ölçmenin Standart Hatası ($SEM$) ve Güven Aralıkları | Tamamlandı | $s \sqrt{1 - r_{xx}}$, %95 / %99 GA | `tests/test_reliability.py` |
| **E** | E.8 | Feldt (1969) Güvenirlik Güven Aralıkları | Tamamlandı | F-dağılımı dönüşümü | `tests/test_reliability.py` |
| **F** | F.1 | Gruplar Arası Test İstatistikleri (Ort., SS, N) | Tamamlandı | Grup betimsel tablosu | `tests/test_group_analysis.py` |
| **F** | F.2 | Bağımsız $t$-testi, Welch $t$-testi ve Cohen's $d$ | Tamamlandı | Birleşik varyans ile Cohen d | `tests/test_group_analysis.py` |
| **F** | F.3 | Tek Yönlü ANOVA ve Eta-kare ($\eta^2$) | Tamamlandı | F-testi ve etki büyüklüğü | `tests/test_group_analysis.py` |
| **F** | F.4 | KTK Madde Adillik Taraması ($\Delta p$ İndeksi) | Tamamlandı | $p_{ref} - p_{focal}$ ve $z$ testi | `tests/test_group_analysis.py` |
| **G** | G.1-16| 12 Adet İnteraktif Plotly Bilimsel Grafiği | Tamamlandı | Yayın kalitesinde akademik grafikler | `ui/charts.py` |
| **H** | H.1 | Çok Sayfalı Stillendirilmiş Excel Raporu (.xlsx) | Tamamlandı | openpyxl, formül korumalı | `tests/test_reports.py` |
| **H** | H.2 | APA 7 Formatında Akademik Word Raporu (.docx) | Tamamlandı | python-docx, APA 7 tablolar | `tests/test_reports.py` |
| **H** | H.3 | CSV Dışa Aktarma Desteği | Tamamlandı | UTF-8-sig formatında | `app.py` |
| **I** | I.1-12| 12 Sayfalı Türkçe Streamlit Kullanıcı Arayüzü | Tamamlandı | Canlıda çalışıyor: `app.py` | `app.py` |
| **E2E**| E2E.1 | Uçtan Uca Bulut Simülasyon Testi | Tamamlandı | 10 aşamalı tam boru hattı | `tests/test_e2e_cloud_simulation.py` |
