import numpy as np
import pandas as pd
from core.irt_multistart_validation import compare_starts

def test_paired_multistart_comparison_same_draw_and_tracks_runtime():
    truth=pd.DataFrame({"Madde":["M1","M2","M3","M4","M5"],   # 4PL needs k>=5
                       "a":[.8,1.1,1.3,1.,1.2],"b":[-.5,.1,.6,-.2,.3],
                       "c":[.08,.12,.1,.11,.09],"d":[.9,.94,.96,.92,.95]})
    result=compare_starts(truth,model="4PL",
                          sample_sizes=(100,),repetitions=1,
                          seed=42,points=11,max_iter=3)
    rows=result["runs"]
    assert len(rows)==2
    assert rows["Tohum"].nunique()==1
    assert set(rows["Strateji"])=={"Tek başlangıç","Üç başlangıç"}
    assert (rows["Süre (sn)"]>=0).all()
    assert rows["Hata"].eq("").all()
    assert len(result["summary"])==2
    assert (result["summary"]["Yakınsama oranı"].between(0,1)).all()
