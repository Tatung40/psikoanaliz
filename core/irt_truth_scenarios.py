"""Explicit truth regimes for IRT Monte Carlo validation.

All regimes use admissible parameter values. These are design scenarios,
not estimates from real student data.
"""
import numpy as np
import pandas as pd

SCENARIOS=("moderate","low_information","extreme_difficulty",
           "high_guessing","lower_upper_asymptote")

def generating_items(model="3PL",k=25,scenario="moderate"):
    if model not in ("3PL","4PL"):
        raise ValueError("Model must be 3PL or 4PL.")
    if not isinstance(k,int) or not 3<=k<=100:
        raise ValueError("Item count must be 3 to 100.")
    if scenario not in SCENARIOS:
        raise ValueError("Unknown simulation scenario.")
    a=np.linspace(.8,1.8,k)
    b=np.linspace(-1.3,1.3,k)
    c=np.linspace(.06,.20,k)
    d=np.ones(k) if model=="3PL" else np.linspace(.87,.97,k)
    if scenario=="low_information":
        a=np.linspace(.35,.75,k)
    elif scenario=="extreme_difficulty":
        b=np.linspace(-2.8,2.8,k)
    elif scenario=="high_guessing":
        c=np.linspace(.24,.32,k)
    elif scenario=="lower_upper_asymptote":
        if model=="4PL":
            d=np.linspace(.68,.80,k)
        else:
            # 3PL has fixed upper asymptote; this regime probes c instead.
            c=np.linspace(.24,.32,k)
    return pd.DataFrame({
        "Madde":[f"M{i+1}" for i in range(k)],
        "a":a,"b":b,"c":c,"d":d})
