# CHANGELOG.md — Sürüm ve Değişiklik Günlüğü

Tüm önemli değişiklikler bu dosyada kronolojik olarak listelenir.

## [1.1.0] - 2026-10-16
### Bulut Dağıtımı Hazırlığı ve İyileştirmeler
- **Linux ve Bulut Uyumluluğu:** Tüm yol yapısı platformlar arası (`os.path.join`) hale getirildi.
- **Tek Tıkla Çevrimdışı Başlatıcı:** Windows kullanıcıları için `START_PSIKOANALIZ.bat` dosyası eklendi.
- **Bulut Yapılandırması:** `.python-version` (3.11) ve `.streamlit/config.toml` optimize edildi.
- **Güvenlik ve Gizlilik:** Kapsamlı `.gitignore` oluşturularak sanal ortamlar, geçici dosyalar ve gizli veriler engellendi.
- **Git Versiyon Kontrolü:** `main` dalı yerel olarak oluşturuldu, tüm kaynak kodlar temiz bir taahhüt (`commit`) ile arşivlendi.
- **Doğrulama:** 30 / 30 birim ve psikometrik test (2.67s) hatasız şekilde doğrulandı.

## [1.0.0] - 2026-10-15
### Tamamlanan ve Doğrulanan Özellikler
- **Core Hesaplama Motoru:** `data_processing`, `scoring`, `test_statistics`, `item_analysis`, `distractor_analysis`, `reliability`, `group_analysis`, `validation`.
- **Arayüz ve Görselleştirme:** 12 interaktif Plotly grafiği ve 12 sekmeli tam Türkçe Streamlit arayüzü (`app.py`).
- **Raporlama:** Çok sayfalı Excel (.xlsx) ve APA 7 Word (.docx) rapor ihracı.
- **Örnek Veri Setleri:** 4 sentetik veri seti oluşturuldu.

## Boyutluluk, yerel bağımsızlık ve madde uyumu ekranları
- `core/dimensionality.py`: Pearson/tetrakorik/polikorik (iki aşamalı ML), KMO, Bartlett, paralel analiz (permütasyon), PAF, varimax/promax, tek boyutluluk özeti.
- `core/local_fit.py`: Yen-Q3, merkezlenmiş Q3, 1F artık korelasyonları, S-X² + BH-FDR, gözlenen/beklenen tabloları. 1PL/2PL asimptotik p/FDR keşifsel (NOT CERTIFIED); 3PL/4PL ve yakınsamayan kestirimlerde p/FDR kapalı; çok kategorili kapalı.
- `ui/dimensionality_localfit_view.py`: iki sol menü ekranı; CSV/XLSX/PNG/ZIP + yöntem/sınırlılık metni.
