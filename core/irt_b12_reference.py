"""B12.4: compare independently obtained, aligned item parameters.

No estimator is run here. The caller must verify matching group scale,
quadrature, identification constraints, bounds and parameterization.
"""
import numpy as np
import pandas as pd

PARAMETERS=("a","b","c","d")

def compare_parameter_tables(python_table, r_table, truth=None, *, group="reference"):
    required={"Madde",*PARAMETERS}
    tables=[pd.DataFrame(python_table).copy(),pd.DataFrame(r_table).copy()]
    if truth is not None: tables.append(pd.DataFrame(truth).copy())
    for table in tables:
        if not required.issubset(table.columns):
            raise ValueError("Missing Madde/a/b/c/d columns")
        if table["Madde"].isna().any() or table["Madde"].duplicated().any():
            raise ValueError("Item IDs must be non-missing and unique")
        for p in PARAMETERS:
            values=pd.to_numeric(table[p],errors="coerce")
            if not np.isfinite(values).all():
                raise ValueError("Parameters must be finite numeric values")
            table[p]=values
    py,r=tables[:2]
    if set(py.Madde)!=set(r.Madde):
        raise ValueError("Python and R item IDs differ: no silent partial join")
    joined=py[["Madde",*PARAMETERS]].merge(r[["Madde",*PARAMETERS]],on="Madde",suffixes=("_py","_r"),validate="one_to_one")
    if truth is not None:
        t=tables[2]
        if set(t.Madde)!=set(py.Madde):
            raise ValueError("Truth item IDs differ")
        joined=joined.merge(t[["Madde",*PARAMETERS]],on="Madde",validate="one_to_one")
        joined=joined.rename(columns={p:f"{p}_truth" for p in PARAMETERS})
    detail=[]
    for _,row in joined.iterrows():
        for p in PARAMETERS:
            v=float(row[f"{p}_py"]); ref=float(row[f"{p}_r"])
            rec={"group":group,"Madde":row["Madde"],"parameter":p,"python":v,"r":ref,"delta":v-ref,"abs_delta":abs(v-ref)}
            if truth is not None:
                actual=float(row[f"{p}_truth"])
                rec.update(truth=actual,python_error=v-actual,r_error=ref-actual)
            detail.append(rec)
    df=pd.DataFrame(detail)
    summaries=[]
    for p,grp in df.groupby("parameter",sort=False):
        rec={"group":group,"parameter":p,"n":len(grp),
             "max_abs_difference":float(grp.abs_delta.max()),
             "mae_python_vs_r":float(grp.abs_delta.mean()),
             "rmse_python_vs_r":float(np.sqrt(np.mean(grp.delta**2)))}
        if truth is not None:
            for name in ("python","r"):
                e=grp[f"{name}_error"]
                rec[f"bias_{name}"]=float(e.mean())
                rec[f"rmse_{name}"]=float(np.sqrt(np.mean(e**2)))
        summaries.append(rec)
    return df,pd.DataFrame(summaries)
