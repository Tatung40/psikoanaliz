# NEXT_STEPS.md — Öncelikli Görevler ve Dağıtım Adımları

**Tarih:** 2026-10-16  
**Mevcut Durum:** Yerel hazırlık, testler, `.gitignore`, `.python-version`, `START_PSIKOANALIZ.bat` ve Git taahhüdü tamamlandı.

---

## Bulut Dağıtımı İçin Sıradaki Adımlar (Kullanıcı Onayı Gerektirenler)

1. [x] Loop 1: Proje denetimi ve Linux uyumluluk kontrolü.
2. [x] Loop 2: 30/30 Pytest bilimsel ve fonksiyonel doğrulama testlerinin yürütülmesi.
3. [x] Loop 3: Bulut hazırlığı (`.gitignore`, `.python-version`, relative paths, bellek içi veri akışı).
4. [x] Loop 7: Windows için tek tıkla başlatıcı (`START_PSIKOANALIZ.bat`) oluşturulması.
5. [ ] **Kullanıcı Onayı & Yetkilendirme:**
   - GitHub deposunun oluşturulması: `https://github.com/<kullanici>/psikoanaliz`
   - Depo görünürlük tercihi (Public vs. Private):
     - *Public (Önerilen):* Dünyadaki tüm kullanıcılar bağlantı üzerinden giriş yapmadan erişebilir.
     - *Private:* Yalnızca davet edilen Google/e-posta hesapları giriş yaparak erişebilir.
   - Kodların GitHub'a push edilmesi için onay/kimlik doğrulama.
6. [ ] **Streamlit Community Cloud Dağıtımı:**
   - [share.streamlit.io](https://share.streamlit.io/) üzerinden uygulamanın bağlanması.
   - `psikoanaliz.streamlit.app` alan adının tanımlanması.
7. [ ] **Canlı Doğrulama:**
   - Canlı URL'nin açılması, test veri setiyle uçtan uca kontrolü ve mobil uyumluluğunun doğrulanması.
