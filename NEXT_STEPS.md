# NEXT_STEPS.md — Tamamlanan İşler ve Gelecek Geliştirme Önerileri

**Tarih:** 2026-10-15  
**Mevcut Durum:** PsikoAnaliz v1.0.0 başarıyla inşa edildi, doğrulandı ve `http://localhost:8501` üzerinde yayında.

---

## Tamamlanan Temel Görevler

1. [x] Python 3.11 ve bağımsız sanal ortam (`.venv`) kurulumu.
2. [x] Tüm açık kaynaklı, sıfır maliyetli kütüphanelerin (`requirements.txt`) yüklenmesi.
3. [x] Kapsamlı CTT istatistik çekirdeğinin (`core/`) yazılması ve test edilmesi:
   - Veri okuma, kayıp veri ve otomatik ayırıcı tespiti.
   - İkili ve Likert puanlama, kuramsal ters puanlama.
   - Test düzeyi istatistikler ve standart $z/T$ puanları.
   - Madde güçlüğü, $r_{it}$, düzeltilmiş $r_{ir}$, $r_{pbis}$, $r_{bis}$ (normal ordinat bazlı), Kelley %27 $D$.
   - Çeldirici analizi, çalışan/çalışmayan/ters çeldirici taraması.
   - Güvenirlik: Cronbach Alfa, KR-20 (Alfa ile kesin eşitlik kanıtı), KR-21, İki-yarı, Omega, SEM, Feldt GA.
   - Grup karşılaştırmaları: $t$-testi, Welch $t$, Cohen's $d$, ANOVA, KTK madde adillik $\Delta p$ taraması.
4. [x] 4 adet sentetik bilimsel örnek veri setinin üretilmesi (`sample_data/`).
5. [x] 12 adet interaktif Plotly grafiğinin (`ui/charts.py`) hazırlanması.
6. [x] Çok sayfalı Excel (.xlsx) ve APA 7 uyumlu Word (.docx) raporlama motorunun inşası.
7. [x] 30 / 30 birim ve psikometrik doğrulama testinin (`pytest`) hatasız geçmesi.
8. [x] 12 sekmeli tam Türkçe Streamlit arayüzünün (`app.py`) başlatılması ve canlı doğrulanması.

---

## Gelecek Sürümler İçin Opsiyonel Geliştirmeler

- [ ] Madde Tepki Kuramı (MTK / IRT) 1PL (Rasch), 2PL ve 3PL modellerinin eklenmesi.
- [ ] Yetenek düzeyine göre koşullu Mantel-Haenszel ve Lojistik Regresyon DIF (Değişen Madde Fonksiyonlaşması) modülü.
- [ ] Çok Boyutlu Açımlayıcı ve Doğrulayıcı Faktör Analizi (AFA / DFA) raporlama sayfası.
- [ ] Kullanıcı isteği doğrultusunda GitHub ve Streamlit Community Cloud üzerine canlı dağıtım (Deployment).
