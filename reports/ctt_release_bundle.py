"""Reproducible KTK report bundle with explicit scientific limitations."""
import io
import zipfile
import pandas as pd
from core.ctt_dual_engine import item_guidance
from reports.ctt_guided_docx import ctt_docx

def _safe(table):
    result=table.copy()
    for c in result:
        if pd.api.types.is_object_dtype(result[c]) or pd.api.types.is_string_dtype(result[c]):
            result[c]=result[c].map(lambda v:"'"+v if isinstance(v,str) and v.lstrip().startswith(("=","+","-","@")) else v)
    return result

def report_tables(result):
    summary=pd.DataFrame([
        {"İstatistik":"N","Yerel değer":result["N"],"Bağımsız değer":None,"Durum":"Tam yanıtlar"},
        {"İstatistik":"Madde sayısı","Yerel değer":result["k"],"Bağımsız değer":None,"Durum":"Analiz edildi"},
        {"İstatistik":"Cronbach alfa","Yerel değer":result["native_alpha"],
         "Bağımsız değer":result["reference_alpha"],
         "Durum":"KARŞILAŞTIRILDI" if result["reference_alpha"] is not None and result["difference"]<=.00002 else "İNCELENMELİ"},
        {"İstatistik":"McDonald omega (yerel PCA yaklaşımı)",
         "Yerel değer":result["native"].get("mcdonald_omega"),
         "Bağımsız değer":None,"Durum":"Bağımsız omega doğrulaması yapılmadı"}
    ])
    return {"Özet":summary,"Madde Önerileri":item_guidance(result),
            "Madde Karşılaştırma":result["item_crosscheck"],
            "İki Yarı Karşılaştırma":result["split_comparison"]}

def export_ktk_xlsx(result):
    buffer=io.BytesIO()
    with pd.ExcelWriter(buffer,engine="openpyxl") as excel:
        for name,frame in report_tables(result).items():
            _safe(frame).to_excel(excel,sheet_name=name[:31],index=False)
    return buffer.getvalue()

def export_ktk_zip(result):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("KTK_Akademik_Rapor.docx",ctt_docx(result))
        z.writestr("KTK_Tablolar.xlsx",export_ktk_xlsx(result))
        for name,frame in report_tables(result).items():
            z.writestr(name.replace(" ","_")+".csv",_safe(frame).to_csv(index=False).encode("utf-8-sig"))
        z.writestr("YONTEMSEL_NOT.txt",("Kararlar uzman onayi gerektirir. Uc gucluk ve dusuk r_ir otomatik "
             "madde silme nedeni degildir. Omega PCA yaklasimi faktorel omega yerine gecmez. "
             "Alfa tek boyutluluk veya gecerligi kanitlamaz. Tum motorlar tam veri uzerinden test edilir.\n").encode("utf-8"))
    return buffer.getvalue()
