"""Reconstitute full B12.5 truth from FREE-parameter comparison table.

Implicit fixed constants are model-defined: a=1 in 1PL, c=0 in 1PL/2PL,
d=1 in 1PL/2PL/3PL. DIF-only focal parameters override anchor values.
"""
import argparse
import pandas as pd
from scripts.b12_7_monte_carlo import ITEMS, PARAMS, TYPES, DIF, load_truth

def extract_truth(source,output):
    d=pd.read_csv(source)
    required={"Madde","grup","param","gercek"}
    if not required.issubset(d.columns):raise ValueError("B12.5 comparison columns not found.")
    relevant=d[list(required)].drop_duplicates()
    if relevant.duplicated(["Madde","grup","param"]).any():
        raise ValueError("Inconsistent or duplicate truth records.")
    lookup={(r.Madde,r.grup,r.param):float(r.gercek) for r in relevant.itertuples(index=False)}
    rows=[]
    for group in ("Kadin","Erkek"):
        for item,typ in zip(ITEMS,TYPES):
            row={"Madde":item,"grup":group, "a":1.,"b":None,"c":0.,"d":1.}
            for p in {"1PL":("b",),"2PL":("a","b"),"3PL":("a","b","c"),"4PL":PARAMS}[typ]:
                key=(item,group,p)
                if group=="Erkek" and item not in DIF:key=(item,"Kadin",p)
                if key not in lookup:raise ValueError(f"Missing true free parameter {key}")
                row[p]=lookup[key]
            rows.append(row)
    truth=pd.DataFrame(rows)
    truth.to_csv(output,index=False)
    return load_truth(output)

def main():
    p=argparse.ArgumentParser();p.add_argument("--comparison",required=True);p.add_argument("--output",required=True)
    args=p.parse_args();result=extract_truth(args.comparison,args.output)
    print(f"Validated {len(result)} complete rows for {result.grup.nunique()} groups.")
if __name__=="__main__":main()
