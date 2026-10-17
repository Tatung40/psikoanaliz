"""Monte Carlo precision diagnostics for 3PL/4PL validation.

Reports uncertainty across independent simulation replications, not across
items treated as independent trials. Conditional error statistics exclude
failed or nonconverged estimates and must be read with convergence rate.
"""
import math
import numpy as np
import pandas as pd

def _wilson(successes, trials, z=1.959963984540054):
    if trials<=0:
        return math.nan,math.nan
    p=successes/trials
    den=1+z*z/trials
    center=(p+z*z/(2*trials))/den
    radius=z*math.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/den
    return max(0.,center-radius),min(1.,center+radius)


def monte_carlo_precision(runs):
    """One row per generating/fitted model and N, with empirical MCSE.

    Each row of runs must represent a separate simulated dataset/replication
    for one condition. No fabricated intervals when fewer than two fits pass.
    """
    frame=pd.DataFrame(runs)
    if not {"Model","N","Yakınsama","MAE"}.issubset(frame):
        raise ValueError("Missing simulation columns.")
    keys=[k for k in ("Veri üreten model","Model koşulu","Model","N")
          if k in frame.columns]
    reports=[]
    for identifier,grp in frame.groupby(keys,dropna=False,sort=False):
        if not isinstance(identifier,tuple):
            identifier=(identifier,)
        n=len(grp)
        status=grp["Yakınsama"].fillna(False).astype(bool)
        accepted=grp.loc[status]
        m=len(accepted)
        lo,hi=_wilson(m,n)
        values=pd.to_numeric(accepted["MAE"],errors="coerce").dropna()
        if len(values)>1:
            mcse=float(values.std(ddof=1)/math.sqrt(len(values)))
        else:
            mcse=math.nan
        row=dict(zip(keys,identifier))
        row.update({"Tekrar":n,"Yakınsayan":m,
                    "Yakınsama oranı":m/n,
                    "Yakınsama %95 Wilson alt":lo,
                    "Yakınsama %95 Wilson üst":hi,
                    "MAE (yakınsayan)":float(values.mean()) if len(values) else math.nan,
                    "MAE MCSE":mcse,
                    "MAE hesaplanan tekrar":len(values)})
        reports.append(row)
    return pd.DataFrame(reports)
