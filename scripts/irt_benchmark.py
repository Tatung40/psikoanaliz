"""Run opt-in IRT parameter-recovery benchmarks from the command line.

Examples:
 python scripts/irt_benchmark.py --sample-sizes 200 --repetitions 1 --max-iter 15
 python scripts/irt_benchmark.py --sample-sizes 200,500,1000,2000 --repetitions 10
The output contains empirical evidence only; no accuracy claims are hard-coded.
"""
import argparse
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.irt_simulation_reporting import (run_timed_recovery, export_recovery_excel,
                                            annotate_generating_model)


def reference_items(k=25, generating_model="4PL"):
    """Moderate interior 4PL generating values; not fitted to real students."""
    if k<3:
        raise ValueError("At least three items required")
    if generating_model not in ("3PL","4PL"):
        raise ValueError("Generating model must be 3PL or 4PL")
    return pd.DataFrame({
        "Madde":[f"M{i+1}" for i in range(k)],
        "a":np.linspace(.8,1.8,k),
        "b":np.linspace(-1.3,1.3,k),
        "c":np.linspace(.06,.20,k),
        "d":np.ones(k) if generating_model=="3PL" else np.linspace(.87,.97,k)})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-sizes",default="200",help="Comma-separated N")
    p.add_argument("--models",default="3PL,4PL")
    p.add_argument("--generating-model",choices=("3PL","4PL"),default="4PL")
    p.add_argument("--repetitions",type=int,default=1)
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--items",type=int,default=25)
    p.add_argument("--max-iter",type=int,default=15)
    p.add_argument("--points",type=int,default=21)\n    p.add_argument("--starts",type=int,choices=(1,3),default=1)
    p.add_argument("--output-dir",default="artifacts/irt-benchmark")
    args=p.parse_args(argv)
    sizes=tuple(int(s) for s in args.sample_sizes.split(","))
    models=tuple(s.strip() for s in args.models.split(","))
    if (len(set(sizes))!=len(sizes) or len(set(models))!=len(models)
            or args.items>50 or args.items<3 or args.max_iter<1
            or args.points<9 or args.points>81
            or args.repetitions<1 or args.repetitions>50
            or len(sizes)*len(models)*args.repetitions>400):
        p.error("Invalid setting or excessive workload")
    result=run_timed_recovery(
        reference_items(args.items,args.generating_model),models=models,
        sample_sizes=sizes,repetitions=args.repetitions,
        seed=args.seed,points=args.points,max_iter=args.max_iter,\n        starts=(0.,-1.,1.)[:args.starts])
    annotate_generating_model(result,args.generating_model)
    path=Path(args.output_dir)
    path.mkdir(parents=True,exist_ok=True)
    (path/"study.xlsx").write_bytes(export_recovery_excel(result))
    (path/"metadata.json").write_text(
        json.dumps(result["metadata"],indent=2,ensure_ascii=False),
        encoding="utf-8")
    result["summary"].to_csv(path/"summary.csv",index=False)
    result["runs"].to_csv(path/"runs.csv",index=False)
    print(json.dumps({"metadata":result["metadata"],
                      "summary":result["summary"].replace(
                          {np.nan:None}).to_dict(orient="records")},
                     ensure_ascii=False,default=str,indent=2))


if __name__=="__main__":
    main()
