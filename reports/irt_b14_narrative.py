"""B14 reproducible Turkish interpretation with evidence and limitations."""
from datetime import date
from reports.irt_b14_report import quality_report

def guided_report(result):
    model=result["model"];n=result["N"];k=result["k"];labels=result["labels"]
    quality=quality_report(result["coded"],labels,model)
    sparse=quality.loc[quality["Düşük frekans (<5)"]>0]
    nominal=model=="NRM"
    model_desc=("Nominal Tepki Modeli (NRM) seçenekleri sıralı başarı düzeyi kabul etmez."
        if nominal else f"{model} sıralı yanıt kategorilerini modeller; bir üst kategori daha yüksek puanı temsil eder.")
    summary=[
        "# B14 Çok Kategorili MTK — Açıklamalı Analiz Raporu",
        f"**Model:** {model}  \\n**Motor:** {result['engine']}  \\n**Örneklem:** {n} kişi  \\n**Madde sayısı:** {k}  \\n**Kategori sayısı:** {len(labels)}",
        "## 1. Analizin amacı ve kapsamı",
        model_desc+" Analiz, açıkça kodlanan kategorili yanıtlar üzerinde yürütülmüştür.",
        "## 2. Veri kalitesi",
        (f"{len(sparse)} maddede en az bir kategorinin frekansı 5'in altındadır. "
         "Bu maddelerin parametrelerinin kararlılığı ayrıca incelenmelidir."
         if len(sparse) else "Ön frekans denetiminde madde başına beşin altında yanıtlanan kategori görülmemiştir. "
         "Bu, tek başına yeterli örneklem veya iyi uyum kanıtı değildir."),
        "## 3. Kestirim ve sınırlılıklar",
        "Motorun ürettiği parametreler indirilebilir tablolarda sunulmuştur. "
        "Ancak standart hataların bilimsel geçerliği, model-veri uyumu, yerel bağımsızlık, "
        "tek boyutluluk ve yöntemler arası eşdeğerlik bağımsız olarak doğrulanmamıştır. "
        "Bu aşamada bulgular keşifsel olarak değerlendirilmelidir.",
        "## 4. Kullanıcı için karar",
        "Kategoriler doğru kodlandıysa ve model kestirimi tamamlandıysa betimsel incelemeye devam edilebilir. "
        "Bilimsel yayın veya yüksek riskli kişi kararları öncesinde model varsayımları, "
        "yakınsama, parametre belirsizliği ve bağımsız referans sonuçları incelenmelidir.",
        "## 5. Raporlama şablonu",
        f"Bu çalışmada {n} kişinin {k} maddeye verdiği yanıtlar {model} kapsamında "
        f"Python mirt 1.2.0 aracılığıyla keşifsel olarak çözümlenmiştir. "
        "Yanıt kategorilerinin frekansları ve madde düzeyindeki seyrek kategoriler raporlanmıştır. "
        "Parametre kestirimlerinin bağımsız bilimsel doğrulaması tamamlanmadığından "
        "çıkarımsal sonuçlara ilişkin temkinli yorum yapılmıştır."
    ]
    return "\n\n".join(summary)+"\n"
