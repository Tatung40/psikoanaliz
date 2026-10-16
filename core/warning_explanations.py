"""Plain-Turkish explanations and actionable diagnostics for IRT outputs."""
import numpy as np
import pandas as pd

PARAMETERS = {
    "a": ("Ayırt edicilik", "Yanıt olasılığı eğrisinin ne kadar dik olduğunu gösterir. "
          "Üst sınırda olması, verinin daha dik eğim isteyebileceği veya kestirimin kararsız olabileceği anlamına gelir."),
    "b": ("Güçlük / konum", "Maddenin yetenek ölçeğindeki konumudur. ±4'e dayanması, "
          "katılımcı yetenekleri ile maddenin yeterince örtüşmediğini gösterebilir."),
    "c": ("Alt asimptot", "Çok düşük yetenekte doğru yanıtlama olasılığının alt sınırıdır. "
          "0 veya 0,35'e yaklaşması, şans düzeyinin veriden iyi ayrılamadığını gösterebilir."),
    "d": ("Üst asimptot", "Çok yüksek yetenekte doğru yanıtlama olasılığının üst sınırıdır. "
          "Alt sınıra yaklaşması, üst yetenek düzeyinde beklenmedik yanlış yanıtların "
          "modeli zorladığını gösterebilir."),
}


def boundary_report(items, model, a_upper=4.0, d_lower=.65):
    """Return per-item explicit bounds; use same thresholds as fit_irt flags."""
    bounds={"a":(.15,float(a_upper),.02,.05),
            "b":(-4.,4.,.05,.05),
            "c":(0.,.35,.005,.005),
            "d":(float(d_lower),1.,.005,.005)}
    active={"1PL":("b",),"2PL":("a","b"),
            "3PL":("a","b","c"),"4PL":("a","b","c","d")}[model]
    found=[]
    for _,row in pd.DataFrame(items).iterrows():
        for par in active:
            low,high,lower_tol,upper_tol=bounds[par]
            value=float(row[par])
            side=("alt" if value<=low+lower_tol else
                  "üst" if value>=high-upper_tol else None)
            if side is not None:
                title,description=PARAMETERS[par]
                found.append({"Madde":row["Madde"],"Parametre":par,
                              "Anlamı":title,"Kestirim":value,
                              "İzin verilen alt":low,"İzin verilen üst":high,
                              "Yaklaşılan sınır":side,
                              "Açıklama":description})
    return pd.DataFrame(found,columns=[
        "Madde","Parametre","Anlamı","Kestirim",
        "İzin verilen alt","İzin verilen üst",
        "Yaklaşılan sınır","Açıklama"])


def opg_explanation(reason,n,k,model,a_upper=4.0,d_lower=.65):
    """Contextual OPG warning with the actual parameter count and policy."""
    per={"1PL":1,"2PL":2,"3PL":3,"4PL":4}[model]
    count=k*per
    if "parametre sayısı" in reason or "örneklem yetersiz" in reason:
        return (
            f"**Neden?** {model} modelinde {k} madde × {per} serbest parametre = "
            f"**{count} parametre** bulunmaktadır. Mevcut OPG güvenlik kontrolü "
            f"en fazla 120 parametreye ve N > 5 × parametre sayısı koşuluna izin verir. "
            f"Sizin verinizde N={n}, gerekli eşik N>{5*count}. "
            "**Bu eşik evrensel bir örneklem kuralı değil, yazılımın temkinli "
            "hesaplama filtresidir.**\n\n"
            "**Ne etkilenir?** Madde parametreleri hesaplanmış olabilir; ancak "
            "yaklaşık standart hata ve %95 güven aralığı üretilememiştir. "
            "SH olmadan belirsizlik hakkında kesin yorum yapılmamalıdır.\n\n"
            "**Ne yapabilirsiniz?** Daha büyük ve uygun örneklem, daha yalın "
            "model (ör. 2PL) veya bağımsız yazılımda doğrulanmış SH yöntemi değerlendirin. "
            "Eşiği sırf sonuç almak için gevşetmeyin."
        )
    if "sınır" in reason:
        return ("**Neden?** En az bir kestirim optimizasyonun izin verdiği "
                "sınıra çok yakındır. Böyle durumlarda normal yaklaşımlı OPG "
                "standart hataları yanıltıcı olabilir.\n\n"
                "**Ne yapabilirsiniz?** Üstteki parametre-sınır tablosunu inceleyin; "
                "model seçimini, örneklemin kapsamını ve bağımsız doğrulamayı değerlendirin. "
                f"Bu kalibrasyonda a üst sınırı {a_upper}, d alt sınırı {d_lower}.")
    if "koşullan" in reason or "tekil" in reason:
        return ("**Neden?** Bilgi matrisi sayısal olarak tekil veya kötü koşullanmış. "
                "Parametre etkileri birbirinden yeterince ayrılamıyor olabilir.\n\n"
                "**Ne etkilenir?** SH ve güven aralığı güvenilir hesaplanamaz. "
                "**Ne yapabilirsiniz?** Aşırı ilişkili/az bilgi taşıyan maddeleri "
                "ve model karmaşıklığını inceleyin; daha fazla veri ya da "
                "bağımsız yöntemle doğrulamayı düşünün.")
    return ("**Açıklama:** "+reason+"\n\n"
            "Bu uyarı madde parametrelerinin otomatik olarak geçersiz "
            "olduğu anlamına gelmez; yalnızca OPG belirsizlik ölçüsü "
            "üretilememiştir. Sonuçları bağımsız yöntemle kontrol edin.")


GUIDE = {
    "yakinsama": (
        "**Nedir?** EM algoritmasının son iki olasılık değerindeki değişim "
        "belirlenen durdurma ölçütünü karşılamamış olabilir.\n\n"
        "**Etkisi:** Parametreler hâlâ değişiyor olabilir; AIC/BIC, madde "
        "uyum p değerleri ve standart hatalar kesin sonuç sayılmamalıdır.\n\n"
        "**Öneri:** Yineleme sayısını artırın, farklı başlangıçları deneyin, "
        "sınır parametrelerini inceleyin ve R mirt ile karşılaştırın."
    ),
    "dortpl": (
        "**Neden?** 4PL her maddede dört parametre kestirir. Özellikle c (alt "
        "asimptot) ve d (üst asimptot) daha çok katılımcı ve farklı yetenek "
        "düzeylerinden yanıt gerektirebilir.\n\n"
        "**Etkisi:** Parametreler sınıra dayanabilir, farklı başlangıçlar "
        "farklı çözümler üretebilir.\n\n"
        "**Öneri:** Yakınsamayı, sınır tanılarını, AIC/BIC'yi ve daha yalın "
        "modellere karşı farkları birlikte inceleyin."
    ),
    "yetenek": (
        "**Nedir?** ML/MAP kişi yetenek kestirimi kullanılan −5 ile +5 "
        "arama ızgarasının ucuna ulaşmıştır. Bu bir madde-parametre sınırı "
        "değildir; kişiye ait θ değerinin arama aralığında sabitlenmesidir.\n\n"
        "**Etkisi:** Gerçek en yüksek olabilirlik noktası aralığın dışında "
        "olabilir; kestirim ve yaklaşık SH dikkatle yorumlanmalıdır.\n\n"
        "**Öneri:** Uç toplam puanları ve madde bilgilerini inceleyin; "
        "EAP gibi önsel kullanan kestirimle duyarlılık kontrolü yapın."
    ),
    "sx2": (
        "**Nedir?** S-X², gözlenen madde yanıtlarını toplam puana koşullu "
        "beklenen yanıtlarla karşılaştırır.\n\n"
        "**Etkisi:** Yakınsamayan veya sınırda kalan modelin p değerleri "
        "kararsız olabilir. p < 0,05 tek başına maddeyi çıkarmak için "
        "yeterli kanıt değildir; çoklu testler de önemlidir.\n\n"
        "**Öneri:** Parametreleri, model yakınsamasını ve madde içeriğini "
        "birlikte değerlendirin."
    )
}
