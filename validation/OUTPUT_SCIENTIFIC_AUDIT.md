# Bilimsel çıktı denetimi — main `45d61f7` (PR #135 sonrası), 2026-11-05

Kapsam: `ACADEMIC_WORD_PYTHON_COVERAGE.md` içindeki 38 analiz rotası ve toplu KTK Word raporu.
Yöntem: Ortak çıktı hattının statik incelemesi (`ui/academic_word.py`, `core/academic_screen_results.py`,
`reports/academic_docx.py`). Şüpheli noktalar küçük deterministik betiklerle yeniden üretildi.
Model kalibrasyonu, simülasyon veya tam regresyon yapılmadı. Kodda değişiklik yapılmadı.

## Özet

| | Sayı |
|---|---|
| Denetlenen rota | 38 (ortak Word/ekran hattı); derin kontrol: KTK Word, Gözlenen Puan DIF, RSM, B14, Python mirt, Boyutluluk/Yerel uyum |
| P0 | 0 |
| P1 | 2 (CONFIRMED) |
| P2 | 2 (CONFIRMED) |

## Bulgular

| # | Öncelik | Durum | Yer | Beklenen | Gerçek | Bilimsel etki |
|---|---|---|---|---|---|---|
| 1 | P1 | CONFIRMED | `reports/academic_docx.py:21–34` (`_section`), `:84–85`; üretici `reports/word_export.py` | Her başlık kendi tablosuyla kalmalı; KTK raporunun 1→5 sırası korunmalı | "4. Klasik Madde Analizi" başlığı ("değerlendirme" kelimesi geçtiği için) Bulgular bölümüne taşınıyor. Tablo 3 (madde güçlük/ayırt edicilik) ise "3. Güvenirlik ve Ölçme Hatası" başlığı altında kalıyor. Bölüm sırası 1, 4, 2, 3, 5 oluyor. | Madde istatistikleri Word'de yanlış başlık altında okunur. Değerlerin kendisi değişmiyor. |
| 2 | P1 | CONFIRMED | `ui/dif_classical_view.py:97`, `ui/irt_python_library_view.py:260`, `ui/dimensionality_localfit_view.py:224, 237, 247`, `ui/irt_rsm_view.py:179` | p < .0001 değerleri "< .001" ya da bilimsel gösterimle yazılmalı | Tablo ekrana gönderilmeden önce `round(4)` (RSM uyumu için `round(3)`) uygulanıyor; Word, ekrana gönderilen bu yuvarlanmış tabloyu yakalıyor. Sonuç: hem ekranda hem Word'de **p = 0**. | Olanaksız bir değer olan "p = 0" raporlanıyor; APA biçimine aykırı. Reddetme kararları değişmiyor (kararlar yuvarlanmamış p'den hesaplanıyor). |
| 3 | P2 | CONFIRMED | `ui/irt_rsm_view.py:145` (`head(500)`), `ui/irt_b14_view.py:140` (`head(200)`) | Tablonun kesildiği belirtilmeli | Ekranda ilk 200/500 kişi gösteriliyor ve bu belirtilmiyor. Word'deki "N satır gösterilmedi" notu yalnız Word'ün kendi 100 satır sınırını sayıyor. | Kişi tablosu eksiksizmiş gibi görünüyor. Tam veri CSV'de mevcut. |
| 4 | P2 | CONFIRMED | `ui/academic_word.py:41–42` (`_frame`: `Styler.data`) | Word'deki biçim ekranla aynı olmalı | Styler biçimlendirmesi Word'e taşınmıyor. Word'de ham değerler 6 anlamlı basamakla (`.6g`) yazılıyor. | Yalnız sunum farkı; değerler aynı. |

## Kontrol edilen ve sorun bulunmayanlar (kanıt)

- **AIC/BIC tutarlılığı:** Python mirt 2PL ve GRM'de, LSAT7 verisiyle (N = 1000, k = 10) AIC = −2LL + 2k ve BIC = −2LL + k·ln N formülleri 1e−8 hassasiyetle tutuyor. `core/irt_rsm.py:215` ve `core/irt_multidim_*` dosyalarındaki formüller de doğru. Yığılmış RSM'deki iki BIC değeri ayrı etiketlerle raporlanıyor (`irt_multidim_rsm_stack.py:80`).
- **NaN/Inf:** `academic_result_schema` dosyasında `allow_nan=False` ve `_json_safe` var; sonlu olmayan değerler Word'de "Raporlanmadı" olarak yazılıyor (`academic_docx.py:195–204`). Toplu raporda eksik değerin 0'a çevrilmediği `tests/test_word_standard.py:98` testiyle doğrulanıyor.
- **Word üretme kapıları:** Yakınsamayan fit, eski veya değişmiş ayarlar ve opak nesneler için Word üretilmiyor (`academic_word.py:184–207`).
- **İndeks ve gizli sütunlar:** İndeks korunuyor (`reset_index`); `column_config=None` ile gizlenen sütunlar Word'e alınmıyor.
- **Önceki denetimlerde doğrulanıp korunanlar:** S-X² p kapıları, sıfır bilgide SEM = NaN, 4PL ML ve 3PL/4PL bootstrap kapıları, DIF-FWER için NOT CERTIFIED etiketleri ve GH101/GH161 kapıları (PR #123–132).

## Doğrulanamayan alanlar (NOT VERIFIED)

- 38 ekranın her birindeki grafiklerin hesaplanan veriyle birebir eşleşmesi (tarayıcıda görsel karşılaştırma yapılmadı).
- Rasch Wright haritası, ICC/IIF/TIF çizimleri, kategori olasılıklarının toplamı ve bilgi fonksiyonunun negatif olmaması (bu çalışmada yeniden test edilmedi; mevcut test paketine dayanılıyor).
- 2PL, DIF ve GRM-DIF Word raporlarında 1 numaralı bulgunun etkisi. Başlık eşleme kuralı aynı, ama bu raporlarda yeniden üretilmedi; incelenen başlık adları doğru eşleniyor.
- MIRT, G Kuramı ve Eksik Veri ekranlarının sayısal içeriği.

## Yeniden üretme

```python
# Bulgu 1
import pandas as pd; from docx import Document; from reports.word_export import create_word_report
f=pd.DataFrame({'Madde':['İ1','İ2'],'N':[5,5],'Güçlük (p)':[.4,.6],'r_it':[.3,.4],'r_ir (Düzeltilmiş)':[.2,.3],
                'D (%27)':[.1,.2],'Silindiğinde Alfa':[.7,.6],'Tanısal Bayrak':['İncele','']})
d=Document(create_word_report({'n_examinees':5,'k_items':2},f,{'cronbach_alpha':.71}))
# Gövde sırası: H2 "4. Klasik Madde Analizi" Bulgular altında; Tablo 3 "3. Güvenirlik ..." altında.
# Bulgu 2
from reports.academic_docx import _value
_value(float(pd.DataFrame({'p':[2.3e-5]}).round(4).p[0]))   # -> '0'
```

## Önerilen düzeltme sırası

1. **Bulgu 2:** p sütunlarını yuvarlamadan bırak; görüntülerken p < .001 için "< .001" yaz (ekran ve Word). p değeri olan yaklaşık 6 tablo.
2. **Bulgu 1:** `_section` kuralı, Heading 1 ile ondan sonraki tabloları birlikte taşımalı; anahtar kelime eşlemesi yalnız kendi tablosu olmayan başlıklara uygulanmalı. KTK raporu için regresyon testi ekle.
3. **Bulgu 3:** Kesilmiş kişi tablolarına "İlk n / N kişi gösteriliyor" notu ekle.
4. **Bulgu 4:** Gerekirse Styler biçimini Word'e taşı.
