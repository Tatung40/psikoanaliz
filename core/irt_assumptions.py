"""IRT preflight: transparent screening, not a replacement for model-based fit tests.

Unidimensionality is screened with a tetrachoric/polychoric-capable future
extension; current Pearson eigenvalue ratio is only an exploratory signal.
Local independence cannot be established from raw responses and is therefore
marked not tested before fitting. Monotonicity is an empirical rest-score
screen (ordinal categories only). No false 'assumption passed' assertions.
"""
import numpy as np
import pandas as pd

def preflight(data,kind="binary",minimum_n=100):
    x=pd.DataFrame(data).apply(pd.to_numeric,errors="coerce")
    n,k=x.shape
    rows=[]
    blocks=[]
    def add(name,status,details,action):
        rows.append({"Varsayım / kontrol":name,"Durum":status,
                     "Açıklama":details,"Öneri":action})
        if status=="ENGEL":blocks.append(name)
    missing=int(x.isna().sum().sum())
    invalid=int((~np.isfinite(x.to_numpy(float))).sum())
    add("Verinin tamlığı","ENGEL" if missing or invalid else "UYGUN",
        f"{missing} eksik/geçersiz yanıt bulundu." if missing or invalid else "Eksik yanıt yok.",
        "Eksik veriler için doğrulanmış bir yöntem seçin." if missing else "Analize devam edilebilir.")
    vals=x.to_numpy(float)
    finite=vals[np.isfinite(vals)]
    coded=bool(len(finite)>0 and (finite>=0).all() and np.equal(finite,np.floor(finite)).all())
    if kind=="binary":coded=coded and np.isin(finite,[0,1]).all()
    add("Kategori kodlaması","UYGUN" if coded else "ENGEL",
        "Yanıt kodları modelle uyumlu." if coded else "Kategori kodlaması bu model için uygun değil.",
        "İkili model için 0/1; sıralı modeller için 0,1,2...; nominal için seçenek kodlarını kullanın.")
    if n<30 or k<3:
        add("Temel boyut","ENGEL",f"N={n}, k={k}; en az 30 kişi, 3 madde gerekir.",
            "Daha fazla yanıt ve madde ekleyin.")
    elif n<minimum_n:
        add("Örneklem büyüklüğü","UYARI",f"N={n}. Karmaşık MTK modellerinde parametreler kararsız olabilir.",
            "Örneklemi büyütün; model karmaşıklığını ve güven aralıklarını inceleyin.")
    else:
        add("Örneklem büyüklüğü","UYGUN",f"N={n}. Bu sayı tek başına yeterli örneklem kanıtı değildir.",
            "Model karmaşıklığına göre ayrıca güç/kararlılık kontrolü yapın.")
    if missing or not coded or n<30 or k<3:
        return {"table":pd.DataFrame(rows),"blocked":bool(blocks),
                "decision":"VERİ DÜZELTİLMELİ","explanation":
                "Veri kodlaması, eksik yanıt veya temel boyut sorunu var. Kestirime geçmeden veriyi düzeltin."}
    constant=x.nunique()<2
    add("Sabit maddeler","ENGEL" if constant.any() else "UYGUN",
        f"{int(constant.sum())} madde herkes için aynı yanıtı içeriyor." if constant.any()
        else "Sabit yanıtlı madde yok.",
        "Sabit maddeleri kontrol edin; gerekirse analiz dışında tutun.")
    if constant.any():
        return {"table":pd.DataFrame(rows),"blocked":True,
                "decision":"VERİ DÜZELTİLMELİ","explanation":
                "Bazı maddelerde herkes aynı yanıtı vermiş. Bu maddeler güvenilir MTK parametre kestirimi için uygun değil."}
    cov=np.corrcoef(vals,rowvar=False)
    eig=np.linalg.eigvalsh(cov)[::-1] if np.isfinite(cov).all() else np.array([])
    if len(eig)>=2 and eig[1]>1e-10:
        ratio=float(eig[0]/eig[1])
        signal="İNCELE" if ratio<3 else "ÖN TARAMA"
        explanation=f"Birinci/ikinci korelasyon özdeğeri oranı {ratio:.2f}. Pearson yaklaşımı yalnızca ön taramadır."
    else:
        signal="İNCELE";explanation="Özdeğer oranı kararlı hesaplanamadı."
    add("Tek boyutluluk",signal,explanation,
        "Kesin karar için kategorilere uygun faktör analizi ve mümkünse model temelli boyutluluk kanıtı gerekir.")
    add("Yerel bağımsızlık","MODEL SONRASI",
        "Maddelerin yetenek sabitken birbirinden bağımsız olması beklenir; ham korelasyonlar bunu test etmez.",
        "Önce MTK modelini kestirin, sonra artık korelasyonları/Q3 ve içerik benzerliğini inceleyin.")
    add("Monotonluk","MODEL SONRASI",
        "Sıralı modellerde yetenek arttıkça daha yüksek kategoriye yönelim beklenir. "
        "Nominal NRM seçeneklerinin doğal başarı sırası bulunmaz." if kind!="binary" else
        "İkili modellerde yetenek arttıkça doğru yanıt olasılığı artmalıdır.",
        "Modelin kategori tepki eğrilerini ve madde parametrelerini denetleyin.")
    add("Madde uyumu ve model-veri uyumu","MODEL SONRASI",
        "Madde uyum istatistikleri ve test bilgi fonksiyonları ancak model kestirildikten sonra değerlendirilebilir.",
        "Kestirim sonrasında S-X² veya modele uygun tanıları inceleyin.")
    decision="DİKKATLİ DEVAM" if signal=="İNCELE" else "ÖN DENETİM TAMAMLANDI"
    return {"table":pd.DataFrame(rows),"blocked":False,"decision":decision,
            "explanation":("Ön veri denetimleri tamamlandı. Model kestirimine geçilebilir, ancak bu sonuçlar "
                           "MTK varsayımlarının kanıtlandığı anlamına gelmez. Tek boyutluluk yalnızca "
                           "keşifsel olarak tarandı; yerel bağımsızlık, monotonluk ve model uyumu "
                           "kestirim sonrasında ayrıca değerlendirilmelidir.")}
