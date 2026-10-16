# PROJECT_STATE.md — PsikoAnaliz Proje Durumu

**Son Güncelleme:** 2026-10-16  
**Mevcut Versiyon:** v1.1.0 (GitHub Üzerinde Canlı, Streamlit Cloud Dağıtımına Hazır)  
**Geliştirici:** Doç. Dr. Mehmet Taha Eser  
**GitHub Deposu:** [https://github.com/Tatung40/psikoanaliz](https://github.com/Tatung40/psikoanaliz)  
**GitHub Görünürlüğü:** PRIVATE (Gizli)  
**Hedef Bulut URL:** [https://psikoanaliz.streamlit.app/](https://psikoanaliz.streamlit.app/)  
**Test Durumu:** 31 / 31 Pytest Testi BAŞARILI (0 Hata, 3.65s)  
**En Son Commit:** `9fc7be4` ("test: Bulut simulasyonu ve uctan uca fonksiyonellik testi")

---

## 1. Mimari ve Bulut Dağıtım Durumu

| Bileşen | Dosya Yolu | Durum | Doğrulama & Notlar |
|---|---|---|---|
| **Veri İşleme** | `core/data_processing.py` | Tamamlandı | Bellek içi (in-memory) veri işleme; diskte kalıcı kayıt tutulmaz |
| **Puanlama Motoru** | `core/scoring.py` | Tamamlandı | Çoktan seçmeli 0/1 puanlama, kuramsal ters puanlama (`Max+Min-X`) |
| **Test İstatistikleri** | `core/test_statistics.py` | Tamamlandı | Betimsel test istatistikleri, örneklem varyansı ($ddof=1$), $z/T$ puanları |
| **Madde Analizi** | `core/item_analysis.py` | Tamamlandı | $p, q, r_{it}$, düzeltilmiş $r_{ir}, r_{pbis}, r_{bis}$, Kelley %27 $D$, silindiğinde alfa |
| **Çeldirici Analizi** | `core/distractor_analysis.py` | Tamamlandı | Seçenek frekansları, %27 üst-alt oranları, $D_{opt}$, çalışan/çalışmayan/ters çeldirici |
| **Güvenirlik** | `core/reliability.py` | Tamamlandı | Cronbach Alfa, KR-20 ($\alpha \equiv KR_{20}$ tam sayısal eşitlik kanıtı), KR-21, İki-yarı, Omega, SEM, Feldt GA |
| **Grup Analizleri** | `core/group_analysis.py` | Tamamlandı | Bağımsız $t$-testi, Welch $t$, Levene, Cohen's $d$, ANOVA, KTK adillik $\Delta p$ |
| **Girdi Doğrulama** | `core/validation.py` | Tamamlandı | Sabit madde ($s^2=0$) tespiti, tek boyutluluk, küçük örneklem kontrolleri |
| **UI Bileşenleri** | `ui/components.py` | Tamamlandı | Özel CSS, responsive mobil uyumlu layout, APA 7 tablo bileşenleri |
| **Görselleştirme** | `ui/charts.py` | Tamamlandı | 12 interaktif Plotly grafiği (Mobil ve masaüstü tarayıcı uyumlu) |
| **Raporlama Motoru**| `reports/excel_export.py`, `word_export.py` | Tamamlandı | Bellek içi `BytesIO` üzerinden Excel (.xlsx) ve APA 7 Word (.docx) ihracı |
| **Ana Uygulama** | `app.py` | Tamamlandı | Oturuma özel `st.session_state` izolasyonu; kullanıcılar birbirinin verisini görmez |
| **Çevrimdışı Başlatıcı**| `START_PSIKOANALIZ.bat` | Tamamlandı | Windows kullanıcıları için tek tıkla çevrimdışı/yerel çalıştırma |
| **Bulut Yapılandırması**| `.python-version`, `.streamlit/config.toml` | Tamamlandı | Python 3.11 hedefli, CORS/XSRF uyumlu yapılandırma |
| **Git & Gizlilik** | `.gitignore` | Tamamlandı | Sanal ortamlar, araçlar, önbellekler, gizli anahtarlar ve veri setleri hariç tutuldu |
| **GitHub Uzak Deposu** | `origin -> https://github.com/Tatung40/psikoanaliz.git` | Aktif | `main` dalı senkronize, PRIVATE görünürlük |

---

## 2. Sıfır Bütçe & Gizlilik İlkeleri
- **Maliyet:** 0 TL. GitHub Free + Streamlit Community Cloud Free Tier.
- **Gizlilik:** Öğrenci/katılımcı verileri hiçbir harici API'ye veya veri tabanına kaydedilmez; oturum bittiğinde bellekten silinir.
- **Depo Ayrımı:** `turkmetinlab` projesine kesinlikle dokunulmamış; bağımsız `psikoanaliz` deposu kurulmuştur.
- **Oturum İzolasyonu:** Streamlit her kullanıcıya bağımsız `session_state` açar; eşzamanlı bağlanan kullanıcılar birbirlerinin verilerine erişemez.
