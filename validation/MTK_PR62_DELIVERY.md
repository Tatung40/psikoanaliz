# PR #62 — doğrulanmış ilerleme ve üretim engelleri

Tarih: 24 Ekim 2026. Dal: `validation-real-ktk-20261023`. Ana dala birleştirme yok.
Başlangıç `28272cb`; önceki tamamlanma ifadeleri yerine çalıştırılmış kanıtlar esas alındı.

## Bu değişiklik

- Uygulama değişikliğinde ikili DIF kapısının çalışması yeni CI politikasıdır.
  Eski GRM yönlendirme beklentisi bununla çelişiyordu. Politika korundu;
  uygulama değişikliği ile yalnız GRM değişikliği ayrı test ediliyor.
- Aktif Python mirt adaptörü artık her maddede beyan edilen kategorilerin
  gözlenmesini gerektirir. Tüm veri setinde kategorilerin bulunması yeterli
  sayılmaz. Boş kategori sessizce yeniden kodlanmaz veya kestirime gönderilmez.
- `browser_irt_smoke.py`: kurulu Chromium ve CDP ile gerçek tarayıcı testi.
  Bağımsız üretilmiş 400×8 CSV gerçekten yüklendi; 1PL/2PL/3PL/4PL için
  gerçek kütüphane kestirimi, grafik varlığı ve sekiz gerçek CSV/XLSX indirmesi
  doğrulandı. Dosyalar açılıp sekiz madde, a/b alanları ve sonlu değerler kontrol edildi.
  Bu çalışma AppTest değildir. Çok kategorili veya DIF tarayıcı kapsamı iddia edilmez.
- `library_polytomous_recovery.py`: mevcut bağımsız üreteçlerden, N600/6 madde/
  üç kategori/iki seed ile aktif adaptörde ayrı tek-grup kurtarma pilotu.
  Yerel DIF deneyleri yeniden çalıştırılmadı. Sekiz fit de yakınsadı.

## Bilimsel kanıt (sertifika değil)

`evidence/pr62_20261024/` bütün sonuçları, başarısızlık alanlarını ve kapsamı saklar.
İkili kurtarma JSON'u mevcut başarılı bilimsel CI artifact'inden alındı;
aynı deney yeniden üretilmedi. Kaynak commit/CI bağlantısı JSON'da bulunur.

| Aktif model | Kanıt | Gözlenen RMSE |
|---|---|---|
| 1PL | Mevcut bağımsız ikili üreteç, iki seed | b .0597 |
| 2PL | Aynı CI | a .1408; b .0790 |
| 3PL | Aynı CI | a .5531; b .3628; c .1294 |
| 4PL | Aynı CI | a 2.3894; b .5424; c .1840; d .0866 |
| GRM | Yeni tek-grup pilotu, iki seed | a .1647–.1815; eşikler .1341–.1492 |
| PCM | Yeni tek-grup pilotu | adımlar .0901–.1121; a sabit |
| GPCM | Yeni tek-grup pilotu | a .1012–.1525; adımlar .0962–.1253 |
| NRM | Yeni tek-grup pilotu | kategori eğimleri .1066–.1665; kesişimler .0888–.0919 |

NRM referans kategori katsayıları sıfıra merkezlenir; eksen işareti ilk
madde/kategori kontrastıyla yönlendirilir. Kategoriler başarı sırası değildir.
İki seed geniş hata/kapsam değerlendirmesi değildir. Sonuçlara bakılarak geriye
dönük kabul eşiği uydurulmadı. Özellikle 4PL'nin yakınsaması iyi kurtarma demek
değildir; eğim RMSE'si üretim yeterliliği iddiasını desteklemiyor.

## Korunan altyapı ve gerçek engeller

- Rasch, KTK, eksik veri ve mevcut formel DIF motorları değiştirilmedi.
- Emekliye ayrılmış motor import envanterinde 16 kayıt var; uygulamanın doğrudan
  eski motor import'u yok. Bu, dolaylı bütün tüketicilerin taşındığı anlamına
  gelmez. DIF/araştırma tüketicileri varken topluca silmek güvenli değildir.
- RSM, Python mirt `fit_mirt` genel adaptöründe desteklenmiyor. Önceden doğrulanmış
  yerel sabit-çapalı RSM DIF, genel RSM tek-grup analiz zinciri değildir; ikisi
  eşdeğer gösterilmedi. Tam RSM zinciri hâlâ eksik.
- 3PL/4PL genel kurtarma/kapsam ve sınır çıkarımı tamamlanmış değil. 4PL
  çıkarımsal DIF engeli korunuyor; üretilmemiş bootstrap p değeri yok.
- NRM ağırlıklı etki aralıklarının önceki düşük kapsamı ve genel DIF FDR/kapsam
  görevleri açık; eski sınırlılıklar kaldırılmadı.
- Çok kategorili/DIF gerçek tarayıcı akışı ve tüm modellerin birleşik
  kişi kestirimi/uyum/SH zinciri için eksiksiz üretim kanıtı henüz yok.
- Bağımsız R mirt nihai karşılaştırması ve kullanıcının gerçek veriyle manuel
  uygulaması istek doğrultusunda bu görevin dışında. Yukarıdaki diğer eksikler
  bu iki istisnanın içine gizlenmez; MTK bütünü üretime hazır ilan edilmez.

Hedefli başlangıç testleri: 28 geçti. İlk tam koşu 796 geçti/2 başarısız/2 atlandı: kaldırılan yerel 3PL arayüzünü bekleyen iki test güncel kütüphane sözleşmesine taşındı; ikisi hedefli koşuda geçti. Sayısal toleranslar gevşetilmedi. Tek tam regresyonun kesin sonucu ve son
commit/CI bağlantıları PR #62 teslim kaydında tutulur; ana dala otomatik merge yok.
