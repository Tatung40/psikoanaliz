# Ordinal DIF: koşullama mekanizması ve analitik geçerlilik

Temel: main `c926b3f` (#116–118 birleşmiş). Yeni rastgele veri, model fit'i, R
kalibrasyonu veya tam regresyon yok. Üretim motorları/çıkarım kapıları/etiketler değişmez.
PR118'in iki impact koşulunda 10'ar eşleştirilmiş veri seti; Mantel/LR Holm-FWER .10,
GRM-LR .00 sonuçları **INCONCLUSIVE / NOT CERTIFIED** olarak kalır. Bunlar genel
performans veya kontrol kanıtı değildir. Kaynaklar: DIF_VALIDATION.md,
FOUR_PL_DIF_PRECISION_2026-11-01.md, ordinal_dif_paired/REPORT.md ve kayıtlı GH mekanizması.

## 1. Latent değişmezlik neden puana koşullu değişmezlik değildir?

Yerel bağımsız GRM'de aynı a_j/b_j iki grupta da kullanılsın. D=1,
`p_jk(theta)=logistic[a_j(theta-b_jk)]-logistic[a_j(theta-b_j,k+1)]`, uç kümülatif
olasılıklar 1/0; kategori kodları 0,...,C−1. Böylece `Y_j ⟂ G | theta` doğrudur.
Grupların latent yoğunlukları `f_g(theta)` farklı olabilir.

`R_j=sum_{i≠j}Y_i`, `S=Y_j+R_j` ve `h_r(theta)=P(R_j=r|theta)` olsun.
Yerel bağımsızlık sayesinde, erişilemeyen r için h=0 tanımıyla:

```
N_g(s,k) = ∫ p_jk(theta) h_(s−k)(theta) f_g(theta) dtheta
P(Y_j=k | S=s,G=g) = N_g(s,k) / Σ_l N_g(s,l).
```

Madde parametreleri sabitken bile integral ağırlığı f_g değişir. Genel GRM'de
kategoriye özgü çarpanlar ortak theta çarpanına ayrılamaz; bu oran gruptan bağımsız
olmak zorunda değildir. Bayes biçimi aynı noktayı gösterir:

```
f_g(theta|S=s) ∝ P(S=s|theta) f_g(theta)
P(Y_j=k|S=s,theta) = p_jk(theta)h_(s−k)(theta) / P(S=s|theta).
```

**Hedef S'ye dahilken** son ifade p_jk(theta) değildir. S hem hedefi hem kalan
maddeleri içerir; yerel bağımsızlık, S'ye tekrar koşullandıktan sonra bu bağımlılığı
kaldırmaz. Posterior karışımını doğrudan `∫p_jk f(theta|S)` yazmak bu durumda hatadır.

**Rest-score** için hedef dışarıdadır; yerel bağımsızlık ile:

```
P(Y_j=k|R_j=r,G=g) = ∫p_jk(theta) h_r(theta) f_g(theta)dtheta / ∫h_r(theta)f_g(theta)dtheta.
```

Parça-bütün bağlantısı kalkar, fakat aynı gürültülü rest-score'da latent posteriorlar
hâlâ gruplara göre değişebilir. Değişmez çapalardan oluşan A puanı için de aynı
formül geçerlidir. Doğru çapalar, sonlu çapa puanını hatasız theta yapmaz.

### Yeterlilik ve kapsam

Rasch/uygun birim-eğimli PCM gibi bir-parametreli üstel ailelerde
`P(x|theta)=h(x) exp(theta S−A(theta))`; tam toplam S'ye koşullanınca theta çarpanı
iptal olur. Ortak değişmez maddeler/uygun yerel bağımsızlık altında grup latent
önselleri de bu koşullu dağılımı değiştirmez. Bu **modele özgü** sonuç genel GRM'ye,
rest-score'a veya çapa alt-puanına aktarılamaz. Birim eşit a tek başına bir GRM'yi
Rasch/PCM yapmaz. Pozitif kontrol testi, Rasch tam-toplam yeterliliğini ve rest-score
ayrımını iki sabit theta değerinde, rastgele örneklem olmadan doğrular.

Daha uzun ve bilgi taşıyan bağımsız testler posterioru yoğunlaştırarak etkiyi
azaltabilir; kısa test, tavan/taban, kaba puan eşleştirmesi, kategori sayısı/seyrekliği
ve yük/threshold yapısı sonucu değiştirir. Daha fazla kategori otomatik daha fazla
bilgi değildir. Yerel bağımlılık veya çok boyutluluk, hem konvolüsyon varsayımını
hem standart GRM-LR modelini bozabilir. Normal olmayan latent dağılımı yalnızca
mean/SD ile düzeltmek yeterli olmayabilir. Burada yalnız normal impact0/−.5 incelenir.

## 2. Deterministik karşı örnek: kategori, puan ve sayısal kararlılık

PR118 ile aynı bilinen GRM: J10/C4, a=.9..1.8, b=−1..1+[−1,0,1], referans N(0,1),
odak N(impact,1), tüm madde parametreleri değişmez. Diğer maddelerin puan kütlesi
theta'da polinom konvolüsyonu ile hesaplandı, sonra GH61/GH81 ile integre edildi.
Eski GH koşullu ortalamaları/kütleleri yeniden fit edilmeden doğrulama referansı
olarak kullanıldı; yeni kod tam kategori tablolarını, rest ve çapa puanlarını ekler.

Impact0'da bütün kategori farkları sıfırdır. Impact−.5, **odak−referans**:

| Madde, tam toplam | ΔP(k=0) | ΔP(k=1) | ΔP(k=2) | ΔP(k=3) | Σk·ΔP = ΔE(Y|S) |
|---|---:|---:|---:|---:|---:|
| P09, S=13 | +.016502 | −.010783 | −.005356 | −.000363 | **−.022584** |
| P10, S=16 | +.016793 | −.009157 | −.006909 | −.000727 | **−.025156** |

Dolayısıyla eski .0226/.0252 sayıları **tek bir kategori olasılığı farkı değil**,
koşullu beklenen kodlanmış madde puanı farkının mutlak maksimumudur. Bu puan
hücrelerinin referans/odak olasılıkları P09'da .05088/.05050, P10'da .04993/.04133'tür.
En büyük tek-kategori farkları başka hücrelerdedir: P09 S=11/k=0 için +.017532,
P10 S=13/k=0 için +.018810. Tam grup olasılıkları ve bütün puanlar evidence.json'dadır.

| Eşleştirme | P09 en büyük mutlak ΔE (puan) | P10 en büyük mutlak ΔE (puan) |
|---|---:|---:|
| Tam toplam, 10 madde | .022584 (13) | .025156 (16) |
| Rest, 9 madde | .083271 (22) | .091579 (24) |
| Yalnız P01–P08 çapa puanı | .101191 (21) | .107025 (21) |

Koşullanan olaylar/puan aralıkları ve ağırlıkları farklıdır: bu maksimumlar aynı
stratumda yöntem performansı karşılaştırması veya **rest daha kötü** kanıtı değildir.
Yalnızca rest/çapa eşleştirmesinin sorunu kesin giderdiği iddiasına karşı örnektir.
Tam-toplam uçlarında hedefin cevabı zorunlu olabilir (S=0=>Y=0); R=0 ise hedefi
zorunlu sıfır yapmaz. Bu da koşullama olaylarının neden değiştirilemez olduğunu gösterir.

Eski ortalama farklarıyla maksimum sapma 2.67e-15; GH61/GH81 kategori olasılığı
sapması en çok 1.04e-7. Bu iki-adım kararlılığı bir matematiksel quadrature hata
üst sınırı veya FWER sertifikası değildir. Veri üretimi, optimizasyon veya yeni
p-değeri yok. Bilinen model ve kayıtlı varsayımlar altında deterministik karşı örnek vardır.

## 3. Testlerin gerçek null'ları ve GRM-LR'nin koşullu avantajı

- **Mantel ordinal:** Puan katmanlarında grup ile sayısal ordinal cevap arasındaki
  1-df ilişki/ortalama-trend skorudur. Tam koşullu dağılım eşitliği bir yeterli null'dur;
  test bütün kategori dağılımı farkları için omnibus değildir. Karşıt katman etkileri
  birbirini götürebilir; küçük/sparse katmanlarda chi-square yaklaşımı ayrıca sınırlıdır.
- **Ordinal LR:** `logit P(Y<=k|z,G)=tau_k−beta z−gamma G−delta zG` modelinde
  `gamma=delta=0` ile ikisinin serbest olduğu modelin 2-df karşılaştırmasıdır.
  z, havuzlanmış total/rest puanın standartlaştırılmasıdır. Orantılı odds, doğrusal
  puan etkisi ve yardımcı (nuisance) modelin doğru belirlenmesi ek varsayımlardır; gerçek latent
  GRM'nin puana koşullu karışımı genelde tam bu biçimde değildir.
- **Sabit çapalı GRM-LR:** Latent ölçekte eşit a ve bütün eşikler null'una karşı,
  aynı önceden belirlenmiş değişmez çapalar altında hedef a/eşiklerini serbest bırakır
  (burada 4 df). Odak latent mean/SD'si her hipotezde serbesttir; impact'ı madde
  değişimi diye zorlamaz. Toplam puanı hatasız theta yerine koymaz.

GRM-LR'nin bu estimanda teorik avantajı için doğru ordinal/tek-boyutlu model,
yerel bağımsızlık, doğru grup latent dağılımları, gerçekten değişmez ve yeterli
çapalar, tanımlanabilir ölçek, iç/regular parametreler, yeterli bilgi ve gerçek
nested ML yakınsaması/kararlı quadrature gerekir. Veriyle çapa seçiminin belirsizliği
ayrı problemdir. Pozitif Hessian veya yakınsama bayrağı model uyumu/yerel bağımsızlık
kanıtı değildir; asimptotik LR de sonlu örneklem FWER garantisi vermez.

**Latent null altında gözlenen-puan farkı varsa**, reddetme latent DIF için yanlış
bayrak olabilir; fakat yöntemin gerçekten test ettiği koşullu-ilişki null'u zaten
false olabilir. Bu karşı örnek, o farklı null altında nominal Tip-I kalibrasyonunun
bozulduğunu tek başına kanıtlamaz. Etki büyüklüğü (ör. .025 puan), p-anlamlılığı,
latent-null operasyonel FWER ve model uyumu ayrı niceliklerdir. Farklı df/null'ların
p-değerlerini eşdeğer test veya üstünlük göstergesi gibi karşılaştırmıyoruz.

### Düzeltme adayları

| Aday | Varsayım / beklenen yarar | Yanlılık ve bilimsel sınır | Maliyet |
|---|---|---|---|
| Rest-score | Hedefin mekanik katkısını kaldırır; aynı LR API'si | Hâlâ gürültülü theta; grup posterior farkını gidermez | Çok düşük |
| Çapa puanı | Önceden doğrulanmış invariant çapalar | Kısa/az bilgi veren çapalarda artık impact; seçilmiş çapalar ek belirsizlik | Düşük |
| Grup-impact düzeltmesi | Doğru latent grup modeli ve ölçek bağlantısı | Ham puanı keyfi kaydırmak veya G'yi nuisance yapmak hedef null'u değiştirebilir/gerçek DIF'i örtebilir | Orta; bağımsız doğrulama gerekir |
| Sabit çapalı GRM-LR | Doğru ortak latent model, değişmez çapalar, regular ML | Yanlış dağılım/yerel bağımlılık/çapa hatası ve finite-N riski sürer | Yüksek; mevcut pilotta birkaç saniye/veri |
| Posterior/measurement-error entegrasyonu | Doğru latent model, tanımlama ve belirsizlik yayılımı | EAP noktasına basit eşleştirme hatasız theta değildir; yeni yöntem doğrulanmadan üretime alınamaz | Orta/yüksek |

Bilimsel düzeltme önceliği latent estimandla uyumlu, varsayımları denetlenmiş GRM-LR;
**güvenli olduğu ilan edilmiyor**. En ucuz ayırıcı sonraki deney ise aynı ordinal
LR'de total/rest değişiminin etkisini izole etmektir; rest bir çözüm değil sınama adayıdır.

## 4. Sonraki deney: önkayıt var, yürütme yok

next_experiment_plan.json: aynı N500/500, J10/C4, impact−.5, latent DIF yok;
P09/P10 ortak ailesi, aynı veri üzerinde ordinal LR total vs rest, arındırma yok.
Birincil sonlanım veri düzeyinde Holm olay farkı `D=I_total−I_rest`.
`D in [−1,1] => Var(D)<=1` olduğundan gerçek MCSE<=.02 için en az **2500 tekrar**
aralık-bazlı en kötü durum garantisi verir. Bu bir %95 aralık yarı genişliği .02 hedefi
**değildir**: dağılımdan bağımsız %95 Hoeffding yarı genişliği **.054324** olur.
PR118'in 3632 tekrarlık farklı sonlanım/hassasiyet hedefi değiştirilmedi/tamamlanmadı.

İkincil yöntem-FWER aralıkları %97.5 kesin CP (iki yöntem için ortak >=%95);
pilot yöntem başarısı veya genel doğrulama iddiası yok. Başarısız çiftler silinmez,
p-üretilmeyen analiz sıfır olay sayılmaz; duyarlılık aralığı ve valid-pair sayısı saklanır.
Sadece tam, önceden sabit R'de ve geçerli ailelerde teyit edici karar değerlendirilebilir.

Mevcut iki koşulda LR ortalama süresi ~.2314–.2317 s/veri: iki LR stratejisi için
2500 tekrarda tahmin ~**1158 s (19.3 dk)**; rest maliyeti ölçülmedi, garanti değildir.
Bütçe **1800 s**, önce 30 çift fizibilite; geçerliyse ve sabit R'nin maliyeti bütçeye
sığıyorsa aynı plana devam edilir. Değilse durulur, sonradan eşik/hassasiyet değiştirilmez.
Yön iddiası için önceden tanımlı birincil aralık sıfırı dışlamalıdır; modelin scoped
kontrolü ayrıca üst sınır<=.05 gerektirir. Genelleme ve üretim etiketleri açılmaz.
**Bu deney bu PR'da çalıştırılmadı.**

Yeniden üretim (yalnız konvolüsyon/integrasyon):

```
python -m validation.ordinal_dif_analytic.counterexample --output /tmp/ordinal-counterexample.json
python -m pytest -q tests/test_ordinal_dif_analytic.py
```

Yedi test: bağımsız desen enumerasyonu, impact0 eşitliği, eski cache ile eşleşme,
rest/çapa karşı örneği, GH kararlılığı, Rasch toplam-yeterlilik pozitif kontrolü ve önkayıt/provenance doğrulaması.
Hiçbirinde rastgele yanıt veya kestirim yok. Kod/cache hash'leri, paket sürümleri,
parametreler ve tüm kategori/puan tabloları evidence.json'da; CI/commit son PR kaydında.
