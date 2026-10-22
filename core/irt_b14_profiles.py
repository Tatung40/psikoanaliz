"""Empirical B14 response profiles. These are descriptive, NOT fitted IRT CRCs.

Ordinal: rest score from *other items* bins, avoiding own-item overlap.
Nominal: relative selection frequency by item only; alphabet labels have no
ordinal score or implied person trait.
"""
import numpy as np
import pandas as pd

def empirical_profiles(coded,labels,model,bins=6):
    x=pd.DataFrame(coded)
    cats=len(labels)
    if x.shape[1]<3:raise ValueError("At least three items required")
    rows=[]
    if model=="NRM":
        for item in x:
            counts=x[item].value_counts()
            for j,label in enumerate(labels):
                rows.append({"Madde":str(item),"Kategori":label,"Grup":"Tüm katılımcılar",
                    "Oran":float(counts.get(j,0)/len(x)),"N":len(x)})
        return pd.DataFrame(rows)
    for item in x:
        rest=x.drop(columns=item).sum(axis=1)
        # Rank binning prevents duplicate-score qcut edge problems.
        nbin=min(int(bins),len(x))
        ranked=rest.rank(method="average")
        try:
            groups=pd.qcut(ranked,q=nbin,duplicates="drop")
        except ValueError:
            continue
        for interval,ids in groups.groupby(groups,observed=True).groups.items():
            selected=x.loc[ids,item]
            group_text=f"{rest.loc[ids].mean():.2f}"
            for j,label in enumerate(labels):
                rows.append({"Madde":str(item),"Kategori":str(label),"Grup":group_text,
                    "Oran":float((selected==j).mean()),"N":len(selected)})
    return pd.DataFrame(rows)
