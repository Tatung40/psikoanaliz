import io
import zipfile
import numpy as np
import pandas as pd
from core.ctt_dual_engine import compare_ctt
from reports.ctt_release_bundle import export_ktk_zip,export_ktk_xlsx

def fixture():
    rng=np.random.default_rng(303)
    z=rng.normal(size=(180,1))
    return pd.DataFrame((z+rng.normal(size=(180,9))>0).astype(int),
                        columns=[f"Item{i}" for i in range(9)])

def test_split_comparison_and_export_bundle():
    x=fixture()
    r=compare_ctt(x,list(x))
    splits=r["split_comparison"]
    assert len(splits)==2
    assert splits["r farkı"].max()<0.001
    assert splits["SB farkı"].max()<0.001
    assert zipfile.is_zipfile(io.BytesIO(export_ktk_zip(r)))
    with zipfile.ZipFile(io.BytesIO(export_ktk_zip(r))) as z:
        assert "KTK_Akademik_Rapor.docx" in z.namelist()
        assert "KTK_Tablolar.xlsx" in z.namelist()
        assert "İki_Yarı_Karşılaştırma.csv" in z.namelist()

def test_export_xlsx_reopens():
    r=compare_ctt(fixture(),list(fixture().columns))
    book=pd.ExcelFile(io.BytesIO(export_ktk_xlsx(r)))
    assert "Madde Karşılaştırma" in book.sheet_names
    assert "İki Yarı Karşılaştırma" in book.sheet_names
