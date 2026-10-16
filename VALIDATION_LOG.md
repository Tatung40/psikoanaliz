# VALIDATION_LOG.md — Sayısal Doğrulama ve Psikometrik Formül Kayıtları

**Geliştirici:** Doç. Dr. Mehmet Taha Eser  
**Tarih:** 2026-10-15  

Bu dosya, PsikoAnaliz platformundaki her bir istatistiksel formülün matematiksel tanımını, varsayımlarını, sayısal toleranslarını ve doğrulama protokollerini içerir.

---

## 1. Güvenirlik Formülleri ve Eşdeğerlik Doğrulaması

### 1.1. Cronbach Alfa ($\alpha$)
$$\alpha = \frac{k}{k - 1} \left( 1 - \frac{\sum_{i=1}^k s_i^2}{s_X^2} \right)$$
- $k$: Madde sayısı ($k > 1$).
- $s_i^2$: $i$. maddenin örneklem varyansı ($ddof=1$).
- $s_X^2$: Toplam test puanının örneklem varyansı ($ddof=1$).
- **Varsayım:** Tau-eşdeğerlik (Tau-equivalence) veya en azından tek boyutluluk.

### 1.2. Kuder-Richardson 20 (KR-20)
$$KR_{20} = \frac{k}{k - 1} \left( 1 - \frac{\sum_{i=1}^k p_i q_i}{\sigma_X^2} \right) = \frac{k}{k - 1} \left( 1 - \frac{\frac{N}{N-1} \sum_{i=1}^k p_i q_i}{s_X^2} \right)$$
- İkili ($0/1$) veride, $i$. maddenin örneklem varyansı $s_i^2 = \frac{N}{N-1} p_i (1 - p_i)$'dir.
- **Kritik Eşdeğerlik Kuralı:** Eğer paydada örneklem varyansı $s_X^2$ ($ddof=1$) kullanılıyorsa, pay kısmında da örneklem madde varyansı $\sum s_i^2 = \frac{N}{N-1}\sum p_i q_i$ kullanılmalıdır.
- **Doğrulama Ölçütü:** Kayıp verisiz ikili matriste:
  $$\lvert \alpha - KR_{20} \rvert < 10^{-12}$$

### 1.3. Kuder-Richardson 21 (KR-21)
$$KR_{21} = \frac{k}{k - 1} \left( 1 - \frac{\bar{X} (k - \bar{X})}{k \cdot s_X^2} \right)$$
- Sadece maddelerin eşit güçlükte olduğu varsayımı altında ikili testler için hesaplanır. $KR_{21} \le KR_{20}$ olmalıdır.

### 1.4. İki-Yarı (Split-Half) ve Spearman-Brown
$$r_{SB} = \frac{2 r_{12}}{1 + r_{12}}$$
- $r_{12}$: İki test yarısı arasındaki Pearson korelasyonu.

### 1.5. Guttman İki-Yarı ($\lambda_4$)
$$\lambda_4 = 2 \left( 1 - \frac{s_1^2 + s_2^2}{s_X^2} \right)$$

---

## 2. Madde İstatistikleri Formülleri

### 2.1. Madde Güçlüğü ($p$)
- İkili veride: $p = \frac{\sum X_i}{N}$
- Çok kategorili veride (normalize): $p_{norm} = \frac{\bar{X}_i - Min_{teorik}}{Max_{teorik} - Min_{teorik}}$

### 2.2. Düzeltilmiş Madde-Toplam Korelasyonu ($r_{ir}$ / Madde-Kalan)
$$X_{kalan} = X_{toplam} - X_i$$
$$r_{ir} = \frac{Cov(X_i, X_{kalan})}{s_i \cdot s_{kalan}}$$
- $s_i = 0$ veya $s_{kalan} = 0$ durumunda değer "Tanımsız" (NaN) olarak işaretlenir.

### 2.3. Nokta-Çift Serili Korelasyon ($r_{pbis}$)
$$r_{pbis} = \frac{\bar{X}_1 - \bar{X}_0}{s_X} \sqrt{\frac{N_1 N_0}{N (N - 1)}} = \frac{\bar{X}_1 - \bar{X}_0}{s_X} \sqrt{\frac{N}{N-1} p q}$$
- Pearson korelasyonu formülüyle ($X_i \in \{0, 1\}$ ve $X_{toplam}$) birebir özdeştir.

### 2.4. Çift-Serili Korelasyon ($r_{bis}$)
$$r_{bis} = \frac{\bar{X}_1 - \bar{X}_0}{s_X} \cdot \frac{p \cdot q}{y}$$
- $y = \frac{1}{\sqrt{2\pi}} e^{-\frac{z_p^2}{2}}$ (Standart normal eğrinin $z_p$ noktasındaki ordinatı; $P(Z < z_p) = p$).
- $r_{bis} \approx 1.253 \cdot r_{pbis}$ (yaklaşık ilişki; $p=0.5$ civarında).

### 2.5. Üst-Alt %27 Ayırt Edicilik İndeksi ($D_{27}$)
$$D = p_{üst} - p_{alt} = \frac{R_{üst}}{N_{üst}} - \frac{R_{alt}}{N_{alt}}$$
- Kesme noktalarında eşit puan alan katılımcılar (ties) için frekans ağırlıklı veya kapsayıcı ortalama protokolü uygulanır.

---

## 3. Normallik ve Standart Puan Formülleri

### 3.1. Standart Puanlar
$$z = \frac{X - \bar{X}}{s_X}$$
$$T = 50 + 10 \cdot z$$

### 3.2. Ölçmenin Standart Hatası ($SEM$)
$$SEM = s_X \sqrt{1 - r_{xx}}$$
$$GA_{\%95} = X \pm 1.96 \cdot SEM$$
