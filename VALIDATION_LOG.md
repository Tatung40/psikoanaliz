# VALIDATION_LOG.md — Sayısal Doğrulama ve Psikometrik Formül Kayıtları

**Geliştirici:** Doç. Dr. Mehmet Taha Eser  
**Son Doğrulama Tarihi:** 2026-10-16  
**Test Çalıştırması:** 30 / 30 PASSED (Birim ve Sayısal Doğrulama)

Bu dosya, PsikoAnaliz platformundaki her bir istatistiksel formülün matematiksel tanımını, varsayımlarını, sayısal toleranslarını ve doğrulama protokollerini içerir.

---

## 1. Güvenirlik Formülleri ve Eşdeğerlik Doğrulaması

### 1.1. Cronbach Alfa ($\alpha$)
$$\alpha = \frac{k}{k - 1} \left( 1 - \frac{\sum_{i=1}^k s_i^2}{s_X^2} \right)$$
- $k$: Madde sayısı ($k > 1$).
- $s_i^2$: $i$. maddenin örneklem varyansı ($ddof=1$).
- $s_X^2$: Toplam test puanının örneklem varyansı ($ddof=1$).

### 1.2. Kuder-Richardson 20 (KR-20)
$$KR_{20} = \frac{k}{k - 1} \left( 1 - \frac{\frac{N}{N-1} \sum_{i=1}^k p_i q_i}{s_X^2} \right)$$
- **Doğrulama Sonucu:** `test_kr20_and_cronbach_alpha_exact_equivalence`
  $$\lvert \alpha - KR_{20} \rvert < 10^{-12}$$
  Örneklem varyansı esasına göre hesaplanan KR-20, Cronbach Alfa ile tam sayısal özdeşlik gösterir.

### 1.3. Kuder-Richardson 21 (KR-21)
$$KR_{21} = \frac{k}{k - 1} \left( 1 - \frac{\bar{X} (k - \bar{X})}{k \cdot s_X^2} \right)$$
- `test_kr21_less_than_or_equal_to_kr20` ile $KR_{21} \le KR_{20}$ eşitsizliği doğrulanmıştır.

### 1.4. İki-Yarı (Split-Half) ve Spearman-Brown
$$r_{SB} = \frac{2 r_{12}}{1 + r_{12}}$$

### 1.5. Guttman İki-Yarı ($\lambda_4$)
$$\lambda_4 = 2 \left( 1 - \frac{s_1^2 + s_2^2}{s_X^2} \right)$$

---

## 2. Madde İstatistikleri Formülleri

### 2.1. Madde Güçlüğü ($p$)
- İkili veride: $p = \frac{\sum X_i}{N}$
- Çok kategorili veride: $p_{norm} = \frac{\bar{X}_i - Min_{teorik}}{Max_{teorik} - Min_{teorik}}$

### 2.2. Düzeltilmiş Madde-Toplam Korelasyonu ($r_{ir}$)
$$X_{kalan} = X_{toplam} - X_i$$
$$r_{ir} = \frac{Cov(X_i, X_{kalan})}{s_i \cdot s_{kalan}}$$

### 2.3. Nokta-Çift Serili ($r_{pbis}$) ve Çift-Serili Korelasyon ($r_{bis}$)
$$r_{bis} = \frac{\bar{X}_1 - \bar{X}_0}{s_X} \cdot \frac{p \cdot q}{y}$$
- $y$: Standart normal eğrinin $z_p$ noktasındaki ordinatıdır.

### 2.4. Üst-Alt %27 Ayırt Edicilik İndeksi ($D_{27}$)
$$D = p_{üst} - p_{alt}$$
- Yüzdelik kesme değerlerine göre tie (bağ) yönetimi ile istikrarlı gruplama.

---

## 3. Bulut Dağıtım Uyumluluk Doğrulaması

1. **İşletim Sistemi Bağımsızlığı:** Windows ve Linux (Debian tabanlı Streamlit Cloud) üzerinde test edilmiştir; `os.path.join` kullanılmıştır.
2. **Bellek İçi Raporlama:** `openpyxl` ve `python-docx` dosya sistemine yazmadan `io.BytesIO` ile bellek üzerinden kullanıcıya indirilir.
3. **Oturum İzolasyonu:** Streamlit `st.session_state` mimarisi sayesinde her tarayıcı sekmesi izole çalışır.
