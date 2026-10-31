# Bilimsel üretim hazırlığı — risk temelli denetim, 2026-10-31

Başlangıç: `24c12e2`; #115 entegrasyonu `93bac01` ayrıca korunur.
Envanter: `app.py` yönlendirmeleri ve `core.example_datasets.analysis_catalog()`:
**39 analiz girdisi** (aynı motorun birden fazla ekranı var). Aşağıdaki **23 değerlendirme
birimi**, ekran veya bağımsız model sayısı değildir. Henüz olmayan DFA/regresyon
modülleri eklenmiş sayılmaz. Açık #65/#69/#72/#74/#83/#100/#20/#23 dalları değiştirilmedi.

## Bilimsel karar matrisi

K = **KOŞULLU DOĞRULANDI**: sadece belirtilen özellik ve mevcut çalışma koşulları;
D = **DOĞRULANMADI**: bu kapsam için yeterli kanıt yok veya başarısız koşul var.
Hiçbiri bütün veri kümelerine/genel üretim sertifikasına karşılık gelmez.

| Analiz | Simülasyon / veri | Referans uyumu | Sayısal kararlılık / açık sınır | Karar | Etiket kararı |
|---|---|---|---|---|---|
| Rasch / sabit-a=1 1PL | p6: 300 tekrar, N1000/J12, normal θ | Phase1: ≤4.42e-4 parametre | b kapsama .953; tekrar-kümeli MCSE tarihsel kayıtta yok; yerel serbest-σ Rasch ayrı model | K | Koşullar kalır |
| Ortak-serbest-a 1PL | Sabit-a kanıtı aktarılamaz | Eşdeğer serbest-σ Rasch matematiği genel kurtarma çalışması değildir | Özel kapsam/GA kanıtı eksik | D | NOT VERIFIED kalır |
| 2PL | p6: 300 tekrar, N1000/J12 | Phase1: ≤.000923 parametre | a/b kapsama .945–.951; kümeli MCSE bilinmiyor | K | Koşullar kalır |
| 3PL | p1: 20/hücre; p7: 100–200/hücre | Eşlenmiş Phase1: ΔLL 1.537e-7, Δparam .0002353 | ML sınırlı; MAP önselleri ayrı; GA desteği parametre/koşula bağlı | K | Kullanım/GA kapıları kalır |
| 4PL | p1: ML kurtarma başarısız; MAP sınırlı | R sıcak başlangıç tanısı: ΔLL .000228813, ΔICC .008999804 | R 3000 döngüde yakınsamadı; ML sınırları, zayıf tanımlanma | D | NOT VERIFIED / NOT CERTIFIED kalır |
| GRM | p4: N500/2000, J10, C4, 50/hücre | `r_mirt_polytomous`: bağımsız R karşılaştırma kaydı | Normal θ; N2000 a/b kapsama .958/.952; seyrek kategori genellemesi yok | K | Veri/koşul notu kalır |
| PCM | Aynı p4 tasarımı | R `gpcm`, a=1 dönüşümü | N2000 adım kapsaması .940; ortak-serbest-a ile karıştırılmaz | K | Koşullar kalır |
| GPCM | Aynı p4 tasarımı | R `gpcm`, kümülatif adım dönüşümü | N2000 a/adım kapsama .952/.966; J/C sınırlı | K | Koşullar kalır |
| NRM | p4 adaptör: 50/hücre | R nominal, referans kategori ve işaret hizalaması | Açık Oakes gerekli; varsayılan SE hatalı; nominal kodlar sıralı başarı değildir | K | Nominal/uyum/Wright kısıtları kalır |
| RSM | p3: 100/hücre, N300/1000/3000, J10/20 | Python alt API 20/20; R RSM karşılaştırma yolu mevcut | N300/J20 δ kapsama .915 başarısız; serbest-σ ve sabit-σ ayrı | K | Küçük örneklem sınırı kalır |
| İkili gözlenen-puan DIF | p2, 100/hücre | MH/RBG statsmodels eşleşmesi | Tip-I .035–.073; BH etiketinin her tasarımda FDR garantisi yok | K | Etki/arındırma notları kalır |
| Sıralı gözlenen-puan DIF | p2, 100/hücre | İstatistik formül kontrolü | Etki altında ordinal LR Tip-I .080, Holm FWER .10–.12'ye çıkıyor | D | Mevcut uyarı kalır |
| Sabit-çapalı formel IRT DIF | Mevcut modele özgü çalışmalar, p2 2PL 40/hücre | Yerel↔kütüphane LR eşleşmesi bağımsız R değildir | Çapa seçimi sonrası çıkarım, geniş FDR tasarımı ve NRM etki GA eksik | K | Sadece incelenen sabit-çapa kapsamı |
| Python mirt çok-gruplu DIF | 12 null tekrar; 2PL tek DIF fikstürü | Bağımsız R yok | Null 1/12; GA [.002108,.384796]; Hessian/SE yok | D | p/q/Wald kapalı kalır |
| 3PL/4PL bootstrap DIF | Mevcut warp-speed 1300 tekrar | Girdi bazlı uygunluk sertifikası yok | 3PL iç c'de boyut .076; 4PL tasarımı dar | D | UI/CSV p değerleri kapalı kalır |
| S-X² madde uyumu | Sabit-parametre referans kontrolü | 1PL/2PL keşifsel; tüm kestirim zinciri değil | 3PL/4PL p/FDR kapalı; çok kategorili yol kapalı | D | REFERENCE CHECKED / NOT CERTIFIED kalır |
| Q3 / merkezlenmiş Q3 | Mevcut artık/yerel bağımlılık testleri | Korelasyon formülleri | Evrensel eşik yok; veri temelli bootstrap ayrı | K | Deneysel çıkarım notu kalır |
| ICC/IRF, IIF/TIF/SEM, EAP | p4 cebir; Phase2 posterior kontrolleri | Bağımsız türev/olasılık ve GH posterior | Koşullu Fisher SEM, posterior SD ve parametre SH farklıdır; diğer kişi yollarında kesik ızgara | K | Kalibrasyon sınırlılığı devralınır |
| KTK, madde/çeldirici, alfa/KR20, grup/rapor | Mevcut deterministik/LSAT6 testleri | Pingouin, statsmodels ve formül çapraz kontrolleri | Alfa boyutluluk kanıtı değildir; veri silme/atama etkisi belirtilmeli | K | Var olan yöntem koşulları kalır |
| Boyutluluk/AFA | Korelasyon, KMO, Bartlett, PAF/rotasyon testleri | SciPy/factor_analyzer ve formül kontrolleri | KMO'nun p değeri yok; Bartlett p ayrı; ordinal/seyrek veri genellemesi yok | K | Varsayım notları kalır |
| G/D kuramı | 5 desen, 15 örnek | Bağımsız EMS ve mevcut R ANOVA CI yolu | Dengeli rastgele yüzeyler; desteklenmeyen desenler kapalı | K | Desen kısıtları kalır |
| MIRT / bifaktör | 2D–5D profil kıyasları; p8 kurtarma başarısız | Modele/profil/kısıta özgü R kayıtları | Bifaktör özgül RMSE .745/.423; genellenebilir kurtarma/GA yok | D | EXPERIMENTAL / NOT CERTIFIED kalır |
| Kayıp veri/atama ve varsayım kararları | Mevcut listwise/atama testleri | İşlev kontrolleri bilimsel MAR/MNAR sertifikası değildir | Her atama için sonuç yanlılığı/kapsama kanıtı burada eşleşmedi | D | Yöntem/varsayım uyarıları kalır |

**Sayım (23 birim): DOĞRULANDI 0; KOŞULLU DOĞRULANDI 15; DOĞRULANMADI 8.**
Bu, var olan bütün işlemlerin hatalı olduğu anlamına gelmez. Genel sertifika verilmedi.
Hiçbir etiket kaldırılmadı; yeni kullanım yetkisi veya çıkarım kapısı açılmadı.

## Doğrulanmış kusur ve minimal düzeltme

`binary_se_coverage_study.py` ve `binary34_se_coverage_study.py` daha önce MCSE'yi
`√(p(1−p)/toplam_aralık)` ile hesaplıyordu. Aynı veri/latent örneklem/kalibrasyondaki
madde aralıkları bağımsız değildir. Dolayısıyla eski gerekçelerdeki “300×12” veya
“100×30 bağımsız aralık” kesinliği gösterilmiş değildir; MCSE aşağı veya yukarı
sapabilir. Nokta kapsama oranları ve önceden kayıtlı eşikler bu kusurla değişmez.

Yeni `coverage_precision.py` bağımsız veri tekrarını küme olarak kullanır. Her tekrarın
kapsanan/uygun sayıları Cᵣ/Mᵣ, p=ΣCᵣ/ΣMᵣ olmak üzere:

`SE = sqrt[R/(R−1) × Σ(Cᵣ−p Mᵣ)²] / ΣMᵣ`.

Bu oran delta-yöntemidir; değişen uygun aralık sayısını korur. Başarısız/SH'siz
tekrarlar sıfır katkıyla ve ayrı başarısızlık oranlarıyla tutulur. Kapsama hâlâ uygun
aralıklara koşulludur. R<2 veya hiç uygun aralık yoksa MCSE `null`; ölçülmüş sıfır
MCSE kesinlik sertifikası değildir. Yeni çıktılar tekrar bazlı Cᵣ/Mᵣ'yi de saklar.
Üretim kestiricileri, tohumlar, önseller, sınırlar, `judge`, eşikler ve CI/GA kapıları değişmedi.

Eski p6/p7 JSON'ları yalnız birleştirilmiş sonuç saklıyor. **Tekrar-kümeli MCSE
bunlardan hesaplanamaz: UNKNOWN.** Ham kanıt ve tarihsel önkayıt gerekçeleri değişmedi;
yeni formül geçmiş sonuçlara uygulanmış gibi gösterilmez. Bu nedenle eski GA
çalışmaları tek başına genel bilimsel sertifika veya etiket kaldırma gerekçesi değildir.

## Yüksek riskli açık işler ve durma ölçütü

1. 4PL: **PARTIAL NUMERICAL AGREEMENT / NOT CONVERGED / NOT CERTIFIED**. Kaydedilmiş
   aktif-yüz Hessian min özdeğer .037891, ölçekli koşul 2349.5; en zayıf yön %99.14 M12.
   Bu pratik zayıf tanımlanma kanıtıdır; yapısal tanımlanamazlık/global optimum kanıtı değildir.
   Yeni fit/Hessian/başlangıç araması yapılmadı.
2. 3PL/4PL kurtarma çalışmasındaki `common_loglik` (`irt_3pl4pl_root_cause.py`)
   `linspace(-6,6,61)` ve normalize normal ağırlık kullanır; bu **GH61 değildir**.
   Kestirimdeki GH düğümleri ve Phase1 kaydedilmiş ortak-GH karşılaştırmasıyla karıştırılmaz.
   Eski sayılar yeniden hesaplanmadı ve değiştirilmedi.
3. p6/p7'nin bağımsız tekrar düzeyindeki belirsizliği eksik. Yeni pahalı çalışma yerine
   önce mevcut ham tekrar çıktısı/artifact aranmalı; yoksa önkayıtlı sınırlı pilotta
   kümeli sayılar saklanmalı. MCSE/kararlılık görülmeden 200–500 tekrara geçilmemeli.
4. Seyrek kategori, normal olmayan θ, otomatik çapa sonrası FDR, NRM kategori-etki GA,
   MIRT kurtarma ve 4PL bağımsız yakınsama ayrı açık kapsamlar olarak kalır.

## Kanıt ve yürütme kaydı

Kaynaklar: `r_mirt_phase1/{README.md,final_summary.json,observed_information/README.md}`,
`r_mirt_polytomous/README.md`, `r_mirt_34/README.md`, `r_mirt_rsm/README.md`,
`POLYTOMOUS_MC_VALIDATION.md`, `RSM_VALIDATION.md`, `DIF_VALIDATION.md`,
`PYTHON_MIRT_DIF.md`, `COVERAGE_MATRIX_2026-10.md`, `PYTHON_LIBRARY_AUDIT_2026-10.md`.
Sayısal kanıtlar: `evidence/p1_3pl4pl`, `p2_dif`, `p3_rsm`, `p4_polytomous`,
`p6_binary_se`, `p7_binary34_se`, `p8_bifactor`. Orijinal dosyalar korunur.

Yerel Rscript yok: **yeni bağımsız R yürütmesi NOT EXECUTED**. Önceki R kayıtları yeni
bir R deneyi sayılmadı. Başlangıç main `24c12e2` üzerinde pytest, Native R no-fit ve
MTK quick/browser kontrolleri başarılıydı (koşular 41206968771, 41206968871);
bu kayıt yeni dalın kabulü değildir. Sonuç/CI ve tam regresyon kayıtları bu değişikliğin
PR'ında SHA ile tutulur; başarısızlık varsa saklanır. Bilimsel açık kapsamlar nedeniyle
platform-geneli üretim sertifikası ve otomatik main birleştirmesi **yok**.
