# PsikoAnaliz — Klasik Test Kuramı Analiz Platformu

**Geliştirici:** Doç. Dr. Mehmet Taha Eser  
**Lisans:** Açık Kaynak (MIT)  
**Maliyet:** 0 TL (Tamamen Açık Kaynak ve Ücretsiz Barındırma Uyumlu)

PsikoAnaliz; eğitimde ve psikolojide ölçme ve değerlendirme uzmanları, psikometristler, öğretmenler, akademisyenler ve lisansüstü araştırmacılar için geliştirilmiş kapsamlı, Türkçe, web tabanlı bir Klasik Test Kuramı (KTK / CTT) analiz platformudur.

Platform, **ShinyItemAnalysis** CTT modülleriyle eşdeğer ve bazı yönleriyle daha gelişmiş analiz, tanısal bayraklandırma, çeldirici analizi, güvenirlik kestirimi, grup karşılaştırmaları, yayın kalitesinde interaktif görselleştirme ve APA 7 uyumlu akademik raporlama imkanı sunar.

---

## 🌟 Temel Özellikler

### 1. Veri İçe Aktarma ve Puanlama (Modül A)
- **Çoklu Format Desteği:** CSV, Excel (.xlsx, .xls) ve SPSS (.sav).
- **Otomatik Algılama:** Karakter kodlaması (UTF-8, ISO-8859-9, Windows-1254) ve ayırıcı algılama.
- **Esnek Puanlama Motoru:**
  - Çoktan seçmeli testler için cevap anahtarı ile otomatik ikili (0/1) puanlama.
  - Önceden puanlanmış (0/1) ikili veri setleri desteği.
  - Çok kategorili (Likert tipi, rubrik, kısmi puan) veri setleri desteği.
  - Teorik alt ve üst sınırlara dayalı ters kodlama (Reverse scoring: `Max + Min - X`).
  - Geçersiz değer ve mükerrer katılımcı kontrolü.
  - Kullanıcı kontrollü kayıp veri yönetimi (Listwise, 0 puanlama, ortalama ikamesi).

### 2. Klasik Madde İstatistikleri (Modül B)
- Madde güçlük indeksi ($p$, $q = 1 - p$), çok kategorili maddeler için normalize güçlük.
- Madde varyansı ve standart sapması.
- Madde-toplam korelasyonu ($r_{it}$) ve **Düzeltilmiş madde-toplam korelasyonu** ($r_{ir}$ / madde-kalan).
- **Nokta-çift serili korelasyon** ($r_{pbis}$) ve **Çift-serili korelasyon** ($r_{bis}$) (normal dağılım ordinatı ile).
- **Üst-Alt %27 Grup Ayırt Ediciliği ($D$):** Eşit puanlar için belgelenmiş hassas bağ (tie) yönetimi.
- Madde silindiğinde test ortalaması, varyansı ve Cronbach alfa katsayısı.
- Otomatik pedagojik/psikometrik tanısal bayraklar (Çok zor, çok kolay, negatif ayırt edicilik, zayıf madde).

### 3. Çeldirici Analizi (Modül C)
- Her seçenek için seçim frekansı ($n$) ve yüzdesi (%).
- Üst %27 ve Alt %27 grupların seçenek seçim oranları.
- Seçenek düzeyinde ayırt edicilik ($D_{seçenek} = \%_{üst} - \%_{alt}$).
- Seçenek-kalan nokta-çift serili korelasyonu.
- Çalışmayan / çekmeyen çeldirici taraması ($< \%5$).
- Ters çalışan çeldirici tespiti (Üst grubu daha çok çeken çeldiriciler).
- Hiç seçilmeyen seçenekler ve potansiyel cevap anahtarı hataları uyarısı.
- İnteraktif seçenek karakteristik eğrileri (Trace lines).

### 4. Test Düzeyi İstatistikler (Modül D)
- Katılımcı sayısı ($N$), Madde sayısı ($k$).
- Ortalama, medyan, mod, minimum, maksimum, ranj.
- Varyans ($s^2$), standart sapma ($s$), ortalamanın standart hatası ($SE_M$).
- Çarpıklık (Skewness) ve Basıklık (Kurtosis).
- Çeyreklikler ($Q_1, Q_2, Q_3$), Çeyrekler arası ranj ($IQR$), Yüzdelikler.
- Standart $z$ puanları, $T$ puanları ve yüzdelik sıralar.
- Taban (Floor) ve Tavan (Ceiling) etkisi analizi.

### 5. Güvenirlik ve Ölçme Hatası (Modül E)
- **Cronbach Alfa ($\alpha$)** ve Standartlaştırılmış Alfa.
- **Kuder-Richardson 20 (KR-20):** İkili verilerde Cronbach alfa ile birebir matematiksel ve sayısal tutarlılık garantisi (örneklem varyans standardı: $ddof=1$).
- **Kuder-Richardson 21 (KR-21):** Eşit güçlük varsayımı altında pratik kestirim.
- **İki-Yarı Güvenirlik (Split-Half):** Tek-Çift, İlk-İkinci Yarı bölmeleri.
- **Spearman-Brown Düzeltmesi** ve **Guttman $\lambda_4$**.
- **McDonald Omega ($\omega_t$):** Tek faktörlü model yükleri üzerinden kuramsal kestirim.
- **Ölçmenin Standart Hatası ($SEM = s \sqrt{1 - r_{xx}}$)** ve Gerçek Puan Güven Aralıkları (%95, %99).
- Feldt (1969) analitik güvenirlik güven aralıkları.

### 6. Grup Bazlı Analizler (Modül F)
- Cinsiyet, okul türü, şube vb. gruplar arası test ve madde istatistikleri karşılaştırması.
- Bağımsız örneklem $t$-testi, Welch $t$-testi ve Cohen's $d$ etki büyüklüğü.
- Çoklu gruplar için Tek Yönlü ANOVA ve $\eta^2$.
- KTK tabanlı madde yanlılığı / adillik taraması ($\Delta p$ analizi).

### 7. Bilimsel Görselleştirme (Modül G)
- Plotly tabanlı, yayın kalitesinde interaktif grafikler:
  1. Toplam Puan Histogramı ve Normal Dağılım Eğrisi
  2. Yoğunluk (Density) Grafiği
  3. Kutu (Boxplot) ve Keman (Violin) Grafiği
  4. Q-Q Normallik Grafiği
  5. Madde Güçlük İndeksi Grafiği (Bölge renklendirmeli)
  6. Madde Ayırt Edicilik Grafiği (Eşik çizgili)
  7. Güçlük - Ayırt Edicilik Saçılım (Kuadrant) Grafiği
  8. Düzeltilmiş Madde-Toplam Korelasyon Grafiği
  9. Madde Silindiğinde Alfa Değişim Grafiği
  10. Çeldirici Seçim Oranları Yığılmış Çubuk Grafiği
  11. Üst-Alt Grup Seçenek Dağılım Çizgileri
  12. Maddeler Arası Korelasyon Isı Haritası
  13. Madde Kalite Matrisi Isı Haritası
  14. Kayıp Veri Matrisi Görseli
  15. Gruplar Arası Puan Dağılım Karşılaştırmaları

### 8. Raporlama ve Dışa Aktarma (Modül H)
- **Excel Raporu (.xlsx):** Çok sayfalı, stillendirilmiş, formüllere karşı güvenli tablo raporu.
- **Word Raporu (.docx):** APA 7 uyumlu tablolar, metodolojik açıklamalar ve yorum yönergeleri içeren akademik rapor.
- **CSV Dışa Aktarma:** Puanlanmış matris, madde istatistikleri ve kestirilen bireysel puanlar.

---

## 🚀 Kurulum ve Yerel Çalıştırma

### Gereksinimler
- Python 3.9, 3.10, 3.11 veya 3.12

### 1. Projeyi Klonlayın veya İndirin
```bash
git clone https://github.com/mehmettahaeser/psikoanaliz.git
cd psikoanaliz
```

### 2. Sanal Ortam Oluşturun (Önerilen)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 3. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 4. Uygulamayı Başlatın
```bash
streamlit run app.py
```
Tarayıcınızda otomatik olarak `http://localhost:8501` adresi açılacaktır.

---

## 🧪 Birim ve Doğrulama Testlerini Çalıştırma

PsikoAnaliz, yüksek psikometrik kesinlik protokolüyle geliştirilmiştir:
```bash
pytest tests/ -v
```

---

## 📂 Proje Yapısı

```text
psikoanaliz/
├── app.py                     # Ana Streamlit uygulaması ve gezinme
├── requirements.txt           # Bağımlılıklar
├── README.md                  # Proje dokümantasyonu
├── PROJECT_STATE.md           # Geliştirme durumu ve bileşenler
├── REQUIREMENTS_MATRIX.md     # Kapsam ve doğrulama matrisi
├── VALIDATION_LOG.md          # Sayısal doğrulama ve formül kayıtları
├── CHANGELOG.md               # Sürüm değişiklik günlüğü
├── NEXT_STEPS.md              # Gelecek geliştirme hedefleri
├── .streamlit/
│   └── config.toml            # Streamlit tema ve sunucu yapılandırması
├── core/                      # İstatistiksel Hesaplama Motoru (UI'dan bağımsız)
│   ├── __init__.py
│   ├── data_processing.py     # Veri okuma, temizleme, kayıp veri
│   ├── scoring.py             # Puanlama, ters kodlama, anahtar eşleştirme
│   ├── test_statistics.py     # Test düzeyi betimsel istatistikler ve normlar
│   ├── item_analysis.py       # Madde güçlüğü, ayırt edicilik, r_pbis, r_bis, D
│   ├── distractor_analysis.py # Çeldirici oranları, D_opt, tanısal uyarılar
│   ├── reliability.py         # Alfa, KR-20, KR-21, İki-yarı, Omega, SEM
│   ├── group_analysis.py      # Grup karşılaştırmaları, t-testi, Cohen d, CTT adillik
│   └── validation.py          # Veri doğrulama ve kontrol kuralları
├── ui/                        # Arayüz ve Görselleştirme Bileşenleri
│   ├── __init__.py
│   ├── components.py          # Akademik kartlar, tablolar, tema
│   └── charts.py              # Plotly grafik şablonları
├── reports/                   # Raporlama Motoru
│   ├── __init__.py
│   ├── excel_export.py        # Excel (.xlsx) ihracı
│   └── word_export.py         # Word (.docx) APA raporu
├── sample_data/               # Örnek Veri Setleri
│   ├── coktan_secmeli_ham.csv # 30 maddelik ham çoktan seçmeli veri
│   ├── ikili_puanlanmis.csv   # 25 maddelik 0/1 puanlanmış veri
│   └── likert_olcek.csv       # 20 maddelik 5'li Likert veri
└── tests/                     # Kapsamlı Pytest Doğrulama Paketi
    ├── __init__.py
    ├── test_scoring.py
    ├── test_items.py
    ├── test_reliability.py
    ├── test_distractors.py
    ├── test_missing_data.py
    ├── test_edge_cases.py
    └── test_group_analysis.py
```

---

## 📑 Bilimsel ve Metodolojik Notlar

### KR-20 ve Cronbach Alfa Sayısal Eşdeğerliği
Klasik Test Kuramı literatüründe (Lord & Novick, 1968; Crocker & Algina, 1986):
Tüm maddelerin 0 ve 1 olarak ikili puanlandığı ve kayıp verinin bulunmadığı durumlarda Cronbach Alfa katsayısı ile Kuder-Richardson 20 (KR-20) katsayısı **özdeştir**:
$$\alpha \equiv KR_{20}$$
PsikoAnaliz, örneklem varyansı ($ddof=1$) standardını hem madde varyanslarında ($s_i^2 = \frac{N}{N-1} p_i q_i$) hem de test toplam varyansında tutarlı olarak kullanır; böylece iki katsayı virgülden sonra en az 12 basamak hassasiyetle birebir eşit çıkar.

### Biserial Korelasyon Formülü
$$r_{bis} = \frac{\bar{X}_1 - \bar{X}_0}{S_X} \cdot \frac{p \cdot q}{y}$$
Burada $y$, standart normal dağılım eğrisinde kümülatif olasılığın $p$'ye eşit olduğu z-değerindeki ordinat (yükseklik) değeridir.

### Üst-Alt %27 Grup Ayırt Ediciliği ($D$) ve Bağ (Tie) Yönetimi
Kelley (1939) teorisine göre normal dağılımda ayırt ediciliği en yüksek duyarlılıkla tespit eden oran %27'dir. PsikoAnaliz, kesme noktalarında eşit puan alan katılımcıları rastgele elemek yerine, frekans ağırlıklı veya kapsayıcı bağ yönetimi uygulayarak istatistiğin istikrarlı olmasını sağlar.

---

## 👨‍🏫 Hazırlayan ve İletişim

**Doç. Dr. Mehmet Taha Eser**  
Eğitimde Ölçme ve Değerlendirme Anabilim Dalı  
Klasik Test Kuramı, Psikometri ve Veri Analitiği  

Sorularınız ve katkılarınız için GitHub Issues üzerinden bildirimde bulunabilirsiniz.
