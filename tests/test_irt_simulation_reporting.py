from io import BytesIO
import pandas as pd
from core.irt_simulation_reporting import run_timed_recovery, export_recovery_excel


def test_timed_simulation_and_excel_export():
    truth=pd.DataFrame({
        "Madde":["M1","M2","M3"],
        "a":[.9,1.1,1.3],"b":[-.5,0.,.5],
        "c":[.08,.1,.12],"d":[.9,.92,.94]})
    result=run_timed_recovery(
        truth,models=("3PL",),sample_sizes=(80,),repetitions=1,
        seed=217,points=11,max_iter=2)
    meta=result["metadata"]
    assert meta["total_model_fits"]==1
    assert meta["elapsed_seconds"]>=0
    assert meta["seed"]==217
    assert len(result["runs"])==1
    payload=export_recovery_excel(result)
    sheets=pd.read_excel(BytesIO(payload),sheet_name=None)
    assert set(sheets)=={
        "Özet","Koşular","Madde_Parametre","Çalışma_Bilgileri"}
    assert len(sheets["Koşular"])==1
    assert "Yakınsama oranı" in sheets["Özet"].columns
    assert "elapsed_seconds" in set(sheets["Çalışma_Bilgileri"]["Ayar"])
