"""B14 reporting exports must preserve counts, labels, and privacy."""
import io
import zipfile
import numpy as np
import pandas as pd
import openpyxl
from reports.irt_b14_report import category_frequencies,quality_report,safe_frame,export_xlsx,export_zip

def pilot():
    x=pd.DataFrame((np.arange(80)[:,None]+np.arange(5)[None,:])%5,columns=list("ABCDE"))
    return x

def test_per_item_category_counts():
    x=pilot();tab=category_frequencies(x,list("ABCDE"))
    assert len(tab)==25
    assert (tab.groupby("Madde")["Frekans"].sum()==80).all()
    assert np.allclose(tab.groupby("Madde")["Yüzde"].sum(),100)

def test_low_frequency_warning():
    x=pilot();x.loc[:77,"A"]=0
    q=quality_report(x,list("ABCDE"),"NRM")
    assert q.loc[q["Madde"]=="A","Uyarı"].iloc[0]=="Kategori seyrekliği"

def test_formula_injection_safe_and_zip_reopens():
    fake={"model":"NRM","engine":"Python mirt 1.2.0","status":"DENEYSEL",
          "N":80,"k":5,"labels":tuple("ABCDE"),"coded":pilot(),
          "result":type("Fit",(),{"converged":True,"log_likelihood":-123.0,
                   "coef_with_se":lambda self:pd.DataFrame({"Parametre":["=2+2"],"Değer":[1.]})})()}
    blob=export_zip(fake)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        assert "B14_rapor.xlsx" in z.namelist()
        wb=openpyxl.load_workbook(io.BytesIO(z.read("B14_rapor.xlsx")))
        assert "Kategori Frekansları" in wb.sheetnames
        assert all(not c.data_type=="f" for ws in wb.worksheets for row in ws for c in row)
        assert "'=2+2" in z.read("parametre.csv").decode("utf-8-sig")
