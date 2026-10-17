"""Opt-in reproducible comparison of single versus three EM starts."""
import argparse
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.irt_benchmark import reference_items
from core.irt_multistart_validation import compare_starts

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--sample-sizes",default="200,500")
    p.add_argument("--repetitions",type=int,default=2)
    p.add_argument("--max-iter",type=int,default=70)
    p.add_argument("--output-dir",default="artifacts/irt-starts")
    args=p.parse_args()
    sizes=tuple(int(v) for v in args.sample_sizes.split(","))
    if (args.repetitions<1 or args.repetitions>10 or len(sizes)>4
            or any(n<30 or n>2000 for n in sizes)
            or args.max_iter<1 or args.max_iter>200):
        p.error("Bounds exceeded")
    directory=Path(args.output_dir)
    directory.mkdir(parents=True,exist_ok=True)
    reports=[]
    for model in ("3PL","4PL"):
        result=compare_starts(
            reference_items(25,model),model=model,
            sample_sizes=sizes,repetitions=args.repetitions,
            seed=2026,points=21,max_iter=args.max_iter)
        result["runs"]["Veri üreten model"]=model
        result["summary"]["Veri üreten model"]=model
        reports.append(result)
    runs=pd.concat([r["runs"] for r in reports],ignore_index=True)
    summary=pd.concat([r["summary"] for r in reports],ignore_index=True)
    runs.to_csv(directory/"runs.csv",index=False)
    summary.to_csv(directory/"summary.csv",index=False)
    with pd.ExcelWriter(directory/"report.xlsx",engine="xlsxwriter") as writer:
        summary.to_excel(writer,sheet_name="Özet",index=False)
        runs.to_excel(writer,sheet_name="Koşular",index=False)
    print(summary.to_string(index=False))

if __name__=="__main__":
    main()
