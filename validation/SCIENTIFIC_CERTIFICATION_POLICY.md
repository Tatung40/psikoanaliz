# Kalıcı Bilimsel Sertifikasyon Politikası — Policy D

**Durum:** Kalıcı proje standardı.  
**Kapsam:** EserPsychometrics'in bütün mevcut modülleri ve gelecekteki geliştirmeleri.

## Temel ilke

“R ile sayısal eşdeğerlik + bilimsel olarak savunulabilir geçerlilik kapsamı + şeffaf metodolojik uyarılar = güvenilir psikometrik yazılım.”

## Policy D — zorunlu kurallar

1. **Bağımsız R referansı:** Sayısal eşdeğerlik, bağımsız R uygulamasıyla yalnızca açıkça test edilmiş veri, model, parametreleme, tanımlama ve kestirim koşullarında değerlendirilir. Sürüm, ayar, dönüşüm ve kaynak kaydedilir. Matematiksel olarak eşdeğer R referansı bulunmadığında sonuç `NOT COMPARABLE` veya `NOT EXECUTED` olarak belirtilir; eşdeğerlik uydurulmaz.
2. **Ayrı sertifikasyon katmanları:** Sayısal uygulama sertifikasyonu; geri kazanım, standart hata, güven aralığı kapsaması, Tip-I hata veya güç gibi kapsamı tanımlı çıkarımsal sertifikasyondan ayrıdır. Sayısal yakınlık, yeşil CI veya başarılı pilot tek başına çıkarımsal geçerlilik kanıtı değildir.
3. **Şeffaf kapsam:** Yöntem varsayımları, uygulanabilirlik koşulları, kestirim türü, sınır çözümleri, yakınsama ve metodolojik sınırlılıklar kullanıcıya ve raporlara açıkça yansıtılır. Sertifika modele bütünüyle değil, belgelenmiş çıktı–yöntem–kapsam birleşimine aittir.
4. **Güvenli çıkarım kapıları:** Desteklenmeyen çıkarımsal iddialar ve bunlara bağlı p-değerleri, güven aralıkları veya kararlar sunulmaz. Bilimsel olarak uygun, sertifikasız tarama/betimsel çıktılar açıkça etiketlenir; çıkarımsal kanıt gibi yorumlanmaz. UI, rapor ve dışa aktarımlar aynı sınırlamaları korur.
5. **Sertifikasyon aklama yasağı:** Bir başka motorun, modelin, kestiricinin, istatistiğin, veri koşulunun veya yalnızca hesaplama özdeşliğinin sertifikası farklı bir iddiaya taşınamaz. Bayesian credible interval, frequentist güven aralığı olarak; MAP sonucu klasik ML çıkarımı olarak yeniden adlandırılamaz. Kısmi sertifika tam sertifika değildir; bastırılmış çıktı sertifikalı sayılmaz.
6. **Kanıt bütünlüğü:** Kabul sınırları sonuçlara bakılarak geriye dönük gevşetilmez. Başarısız, geçersiz, yakınsamayan veya tamamlanmamış çalışmalar gizlenmez, sessizce dışlanmaz ya da başarı gibi sunulmaz. Gerekli yöntem/protokol değişiklikleri bağımsız doğrulama sonuçları görülmeden kaydedilir; eski kanıt korunur.
7. **Yerleşik yöntemlerin sınırları:** Yerleşik istatistiksel yöntemlerin doğal sınırlılıklarını ortadan kaldırmak amacıyla yeni yöntem icat etme zorunluluğu yoktur. Savunulabilir sınırlandırma, açık uyarı ve gerektiğinde çıkarımı kapatma geçerli çözümlerdir; yapay sertifikasyon değildir.
8. **Yeterli kanıtı yeniden kullanma:** Geçerli mevcut referanslar, arşivler, protokoller ve testler yeniden kullanılır. Yalnızca etiketi değiştirmek, kapanışı tekrarlamak veya zaten desteklenen bir iddiayı yeniden üretmek için gereksiz doğrulama kampanyası başlatılmaz.
9. **Yeniden açma koşulları:** Sertifikasyon yalnızca ilgili uygulama, bağımsız referans, varsayım, kapsam, yeni kanıt veya regresyon bulgusu bunu gerektiriyorsa yeniden açılır. Gerekçe ve etkilenen iddialar kaydedilir; ilişkisiz sertifikalar korunur. Kanıt geri çekildiğinde veya geçerlilik kapısı başarısız olduğunda ilgili çıkarım kapatılır.
10. **Verimlilik ve titizlik:** Maksimum bilimsel titizlik; asgari yeterli kanıt, asgari süre ve token tüketimiyle sağlanır. Önce mevcut kanıt incelenir, yalnız ilgili kontroller çalıştırılır. Hesaplama maliyeti doğruluğun önüne geçmez; daha fazla hesaplama da kendiliğinden daha güçlü kanıt sayılmaz.

## Uygulama ve kapanış

Bu politika bütün analiz aileleri, kullanıcı arayüzleri, raporlar, dışa aktarımlar, tekrar üretilebilir kodlar, doğrulama belgeleri ve gelecekteki değişiklikler için geçerlidir. Mevcut sertifikaları, bilimsel eşikleri, kaynak hash'lerini veya analiz algoritmalarını kendiliğinden değiştirmez.

Üretim kapanışı; bilimsel olarak savunulabilir kullanım kapsamı, açık sınırlılıklar, güvenli çıkarım kapıları ve başarılı gerekli yazılım kontrolleriyle değerlendirilir. **Güvenli üretim hazırlığı, bütün olası çıktıların koşulsuz bilimsel sertifikasyonu anlamına gelmez.** Kapsam dışı veya sertifikasız iddialar mevcut sertifikalarla örtülmez.

Yeni bir değişiklikte hangi iddianın etkilendiği ve yeniden doğrulamayı gerektirip gerektirmediği belirtilir. Etkilenmeyen geçerli kanıt korunur; yalnızca gerekli hedefli kontroller ve mevcut zorunlu CI uygulanır. Bu politika tek başına yeni simülasyon veya bilimsel doğrulama kampanyası gerektirmez.
