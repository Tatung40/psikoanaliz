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
