# CHANGELOG.md — Sürüm ve Değişiklik Günlüğü

Tüm önemli değişiklikler bu dosyada kronolojik olarak listelenir.

## [1.0.0] - 2026-10-15
### Tamamlanan ve Doğrulanan Özellikler
- **Python 3.11 ve Sanal Ortam (.venv):** Bağımsız `tools/uv.exe` ile Python 3.11.17 kuruldu ve `.venv` sanal ortamı oluşturuldu.
- **Bağımlılıklar:** Sıfır maliyetli 51 açık kaynak Python kütüphanesi (`streamlit`, `pandas`, `numpy`, `scipy`, `statsmodels`, `plotly`, `openpyxl`, `python-docx`, `pytest`) yüklendi.
- **Core Hesaplama Motoru:**
  - `core/data_processing.py`: CSV, XLSX, SAV okuyucu, otomatik ayırıcı ve kodlama tespiti, 4 kayıp veri protokolü.
  - `core/scoring.py`: Cevap anahtarlı 0/1 ikili puanlama, kuramsal ters puanlama (`Max+Min-X`), toplam puan hesabı.
  - `core/test_statistics.py`: Betimsel test istatistikleri, örneklem varyansı ($ddof=1$), $SE_M$, çarpıklık, basıklık, $z/T$ puanları, frekans tablosu.
  - `core/item_analysis.py`: $p, q, r_{it}$, düzeltilmiş $r_{ir}$, $r_{pbis}$, $r_{bis}$ (normal ordinat bazlı), Kelley %27 $D$ bağ duyarlı ayırt edicilik, silindiğinde alfa, pedagojik bayraklar.
  - `core/distractor_analysis.py`: Seçenek frekansları, %27 üst-alt oranları, $D_{opt}$, seçenek-kalan korelasyonu, çalışmayan ($< \%5$) ve ters çalışan çeldirici taraması.
  - `core/reliability.py`: Cronbach Alfa, KR-20 (Alfa ile kesin eşitlik garantili: $|Alpha - KR20| < 10^{-12}$), KR-21, İki-yarı (Spearman-Brown ve Guttman $\lambda_4$), McDonald Omega, SEM, Feldt %95 güven aralıkları.
  - `core/group_analysis.py`: İki grup için Bağımsız $t$-testi, Welch $t$-testi, Levene varyans testi, Cohen's $d$; 3+ grup için Tek Yönlü ANOVA ve $\eta^2$; KTK madde adillik $\Delta p$ taraması.
  - `core/validation.py`: Sabit madde ($s^2=0$) tespiti, tek boyutluluk, küçük örneklem kontrolleri.
- **Arayüz ve Görselleştirme:**
  - `ui/components.py`: Akademik tema, kurumsal başlık, metrik kartları, APA 7 tablo bileşenleri.
  - `ui/charts.py`: 12 adet interaktif Plotly grafiği (Histogram, Kutu, Q-Q normallik, Güçlük çubuğu, Ayırt edicilik çubuğu, Kuadrant saçılımı, Silindiğinde alfa, Çeldirici oranları, Isı haritası).
- **Raporlama:**
  - `reports/excel_export.py`: Çok sayfalı, stillendirilmiş ve formül enjeksiyonuna karşı korumalı Excel (.xlsx) ihracı.
  - `reports/word_export.py`: APA 7 standartlarında Tablo 1, 2, 3 ve kaynakça içeren akademik Word (.docx) rapor ihracı.
- **Test Paketi:**
  - `tests/`: 30 birim ve psikometrik doğrulama testi yazıldı; tamamı (`30 passed`) 2.04 saniyede hatasız geçti.
- **Uygulama:**
  - `app.py`: 12 sekmeli tam Türkçe Streamlit arayüzü `http://localhost:8501` adresinde canlı ve aktif olarak çalıştırıldı.
