"""B14 descriptive reporting; no unsupported IRT inference is invented."""
import io
import json
import zipfile
import numpy as np
import pandas as pd

def _safe(v):
    if isinstance(v,str) and v.lstrip().startswith(("=","+","-","@")):
        return "'"+v
    return v

def safe_frame(frame):
    out=frame.copy()
    for col in out:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            out[col]=out[col].map(_safe)
    out.columns=[_safe(str(c)) for c in out.columns]
    return out

def category_frequencies(coded,labels):
    rows=[]
    n=len(coded)
    for col in coded:
        counts=coded[col].value_counts()
        for index,label in enumerate(labels):
            count=int(counts.get(index,0))
            rows.append({"Madde":str(col),"Kategori":str(label),
                         "Kod":index,"Frekans":count,"Yüzde":100*count/n})
    return pd.DataFrame(rows)

def quality_report(coded,labels,model):
    rows=[]
    n=len(coded)
    for col in coded:
        counts=coded[col].value_counts()
        proportions=np.array([counts.get(i,0)/n for i in range(len(labels))],float)
        rows.append({"Madde":str(col),"Farklı kategori":int((proportions>0).sum()),
            "En seyrek kategori (n)":int(counts.min()) if len(counts) else 0,
            "En yaygın kategori (%)":float(proportions.max()*100),
            "Düşük frekans (<5)":int(sum(counts.get(i,0)<5 for i in range(len(labels)))),
            "Uyarı":"Kategori seyrekliği" if min(counts.get(i,0) for i in range(len(labels)))<5 else "—"})
    return pd.DataFrame(rows)

def summary_metadata(result):
    raw=result["result"]
    vals={"Model":result["model"],"Motor":result["engine"],
          "Bilimsel durum":result["status"],"N":result["N"],"Madde sayısı":result["k"],
          "Kategori sayısı":len(result["labels"]),"Etiketler":", ".join(result["labels"])}
    for name,attr in (("Yakınsama","converged"),("logLik","log_likelihood"),
                      ("Parametre sayısı","n_parameters"),("Yineleme","n_iterations")):
        try:
            value=getattr(raw,attr)
            if isinstance(value,(str,int,float,bool,np.integer,np.floating,np.bool_)):
                vals[name]=value
        except Exception:
            continue
    return pd.DataFrame([{"Özellik":k,"Değer":v} for k,v in vals.items()])

def report_tables(result):
    frames={"Meta":summary_metadata(result),
        "Kategori Frekansları":category_frequencies(result["coded"],result["labels"]),
        "Kalite Denetimi":quality_report(result["coded"],result["labels"],result["model"])}
    try:
        p=result["result"].coef_with_se()
        if not isinstance(p,pd.DataFrame):p=pd.DataFrame(p)
        frames["Motor Parametreleri"]=p.reset_index()
    except Exception as exc:
        frames["Parametre Açıklaması"]=pd.DataFrame([{"Açıklama":"Motor parametre tablosu mevcut değil.",
                                                       "Gerekçe":str(exc)}])
    return frames

def export_xlsx(frames):
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as writer:
        for name,frame in frames.items():
            safe_frame(frame).to_excel(writer,sheet_name=name[:31],index=False)
    return buf.getvalue()

def export_zip(result):
    frames=report_tables(result)
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("B14_rapor.xlsx",export_xlsx(frames))
        for name,frame in frames.items():
            slug={"Meta":"meta","Kategori Frekansları":"kategori_frekans",
                  "Kalite Denetimi":"kalite","Motor Parametreleri":"parametre",
                  "Parametre Açıklaması":"aciklama"}[name]
            z.writestr(slug+".csv",safe_frame(frame).to_csv(index=False).encode("utf-8-sig"))
        z.writestr("BILIMSEL_UYARI.txt",
            "B14 deneysel Python mirt kestirimi. SH, GA, model uyumu, yerel bagimsizlik "
            "ve parametre esitligi bagimsiz referansla dogrulanmamistir. "
            "NRM secenek kodlari nominaldir; yanlis/dogru ya da basari sirasi degildir.\n")
    return buf.getvalue()
