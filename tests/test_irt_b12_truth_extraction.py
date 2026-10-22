import pandas as pd
from scripts.b12_7_truth_from_validation import extract_truth

def test_sparse_truth_restores_fixed_and_anchors(tmp_path):
    rows=[]
    types=["1PL","2PL","3PL","4PL"]
    from scripts.b12_7_monte_carlo import ITEMS,TYPES,DIF
    for item,typ in zip(ITEMS,TYPES):
        for p in {"1PL":("b",),"2PL":("a","b"),"3PL":("a","b","c"),"4PL":("a","b","c","d")}[typ]:
            rows.append({"Madde":item,"grup":"Kadin","param":p,"gercek":{"a":1.25,"b":0.3,"c":0.16,"d":.87}[p]})
            if item in DIF:rows.append({"Madde":item,"grup":"Erkek","param":p,"gercek":{"a":1.35,"b":.4,"c":.18,"d":.89}[p]})
    src=tmp_path/"comp.csv";dst=tmp_path/"truth.csv"
    pd.DataFrame(rows).to_csv(src,index=False)
    t=extract_truth(src,dst)
    assert len(t)==60 and not t[["a","b","c","d"]].isna().any().any()
    assert t.query("Madde == 'M01' and grup == 'Kadin'").iloc[0]["a"]==1.
    assert t.query("Madde == 'M25' and grup == 'Erkek'").iloc[0]["c"]==.16
    assert t.query("Madde == 'M05' and grup == 'Erkek'").iloc[0]["b"]==.4
