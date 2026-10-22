"""Extract simulation truth from B12.5's externally supplied independent comparison table.

Usage:
python -m scripts.b12_7_truth_from_validation \
 --comparison B12_5/out/compare_item_parameters.csv \
 --output data/b12_7_truth.csv
"""
import argparse
import pandas as pd
from scripts.b12_7_monte_carlo import ITEMS, PARAMS, load_truth

def extract_truth(source,output):
    d=pd.read_csv(source)
    required={"Madde","grup","param","gercek"}
    if not required.issubset(d.columns):raise ValueError("B12.5 comparison columns not found.")
    d=d[list(required)].drop_duplicates()
    if d.duplicated(["Madde","grup","param"]).any():
        raise ValueError("Inconsistent or duplicate truth records.")
    truth=d.pivot(index=["Madde","grup"],columns="param",values="gercek").reset_index()
    truth=truth[["Madde","grup",*PARAMS]]
    truth.to_csv(output,index=False)
    return load_truth(output)

def main():
    p=argparse.ArgumentParser();p.add_argument("--comparison",required=True);p.add_argument("--output",required=True)
    args=p.parse_args();result=extract_truth(args.comparison,args.output)
    print(f"Validated {len(result)} rows for {result.grup.nunique()} groups.")
if __name__=="__main__":main()
