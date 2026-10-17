import math
import pandas as pd
import pytest
from core.irt_monte_carlo_precision import monte_carlo_precision

def test_monte_carlo_uncertainty_is_replication_based():
    trials=pd.DataFrame({
      "Veri üreten model":["4PL"]*4,"Model":["4PL"]*4,
      "N":[200]*4,"Yakınsama":[True,True,False,True],
      "MAE":[.1,.2,.9,.3]})
    row=monte_carlo_precision(trials).iloc[0]
    assert row["Tekrar"]==4 and row["Yakınsayan"]==3
    assert row["Yakınsama oranı"]==.75
    assert row["MAE (yakınsayan)"]==pytest.approx(.2)
    assert row["MAE MCSE"]==pytest.approx(.1/math.sqrt(3))
    assert row["Yakınsama %95 Wilson alt"]<.75<row["Yakınsama %95 Wilson üst"]


def test_single_success_has_no_false_precision():
    x=pd.DataFrame({"Model":["3PL","3PL"],"N":[100,100],
                    "Yakınsama":[True,False],"MAE":[.3,.99]})
    r=monte_carlo_precision(x).iloc[0]
    assert math.isnan(r["MAE MCSE"])
    assert r["MAE hesaplanan tekrar"]==1
    assert r["Yakınsama %95 Wilson alt"]<.5


def test_invalid_simulation_columns_fail_explicitly():
    with pytest.raises(ValueError,match="Missing"):
        monte_carlo_precision(pd.DataFrame({"Model":["3PL"]}))
