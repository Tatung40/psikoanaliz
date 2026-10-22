"""Conservative evidence ledger for B14 analyses: no unsupported claims."""
import numpy as np
import pandas as pd
from reports.irt_b14_report import quality_report

def interpretation_audit(result):
    fit=result["result"]
    quality=quality_report(result["coded"],result["labels"],result["model"])
    sparse=int((quality["Düşük frekans (<5)"]>0).sum())
    converged=getattr(fit,"converged",None)
    rows=[
      {"Alan":"Veri kodlama","Kanıt":"Kategoriler doğrulandı","Durum":"Doğrulandı",
       "Kullanıcıya açıklama":"Yanıtlar seçilen model türüne göre kodlandı. NRM'de seçenekler sırasızdır."},
      {"Alan":"Kategori frekansları","Kanıt":f"{sparse} madde seyrek kategori içeriyor",
       "Durum":"İnceleme önerilir" if sparse else "Betimsel olarak uygun",
       "Kullanıcıya açıklama":"Az gözlenen kategoriler parametre kestirimini kararsızlaştırabilir."},
      {"Alan":"Kestirim yakınsaması","Kanıt":str(converged),
       "Durum":"Başarılı" if converged is True else ("Başarısız" if converged is False else "Doğrulanmadı"),
       "Kullanıcıya açıklama":"Yakınsama, optimizasyonun durma koşulunu gösterir; modelin uygunluğunu tek başına kanıtlamaz."},
      {"Alan":"Tek boyutluluk","Kanıt":"Bağımsız faktör doğrulaması yok",
       "Durum":"Doğrulanmadı",
       "Kullanıcıya açıklama":"Kategorik faktör analizi veya uygun çok boyutlu model karşılaştırması gerekir."},
      {"Alan":"Yerel bağımsızlık","Kanıt":"Model sonrası artık tanısı yok",
       "Durum":"Doğrulanmadı",
       "Kullanıcıya açıklama":"Aynı yetenek düzeyindeki maddeler arası artık ilişki incelenmelidir."},
      {"Alan":"Model/madde uyumu","Kanıt":"Bağımsız uyum doğrulaması yok",
       "Durum":"Doğrulanmadı",
       "Kullanıcıya açıklama":"Uygun uyum ölçüleri ve tanılar doğrulanmadan iyi uyum iddia edilemez."},
      {"Alan":"Parametre SH/GA","Kanıt":"Bağımsız referans karşılaştırması yok",
       "Durum":"Doğrulanmadı",
       "Kullanıcıya açıklama":"Motor çıktısı bulunması tahmin belirsizliğinin doğruluğunu kanıtlamaz."}
    ]
    decision=("ÖNCE KESTİRİM SORUNUNU ÇÖZÜN" if converged is False else
              "KEŞİFSEL İNCELEMEYE DEVAM" if converged is True else
              "YAKINSAMA DURUMUNU DOĞRULAYIN")
    return {"table":pd.DataFrame(rows),"decision":decision,
            "publication_ready":False,
            "explanation":"Bu analiz betimsel/keşifsel inceleme için kullanılabilir; bilimsel yayında güçlü "
            "çıkarımlar veya bireysel kararlar için boyutluluk, yerel bağımsızlık, model uyumu "
            "ve belirsizlik hesapları ayrıca doğrulanmalıdır."}
