import numpy as np
import pandas as pd
import pytest
from core.irt_simulation import repeated_recovery


def _truth():
    return pd.DataFrame({
        "Madde":["M1","M2","M3"],
        "a":[.8,1.2,1.5],"b":[-.6,0.,.7],
        "c":[.08,.12,.14],"d":[.92,.94,.96]})


def test_repeated_recovery_is_reproducible_and_tracks_nonconvergence():
    kw=dict(models=("3PL","4PL"),sample_sizes=(80,),
            repetitions=2,seed=81,points=11,max_iter=2)
    a=repeated_recovery(_truth(),**kw)
    b=repeated_recovery(_truth(),**kw)
    pd.testing.assert_frame_equal(a["runs"],b["runs"])
    pd.testing.assert_frame_equal(a["summary"],b["summary"])
    assert len(a["runs"])==4
    assert set(a["summary"]["Model"])=={"3PL","4PL"}
    assert (a["summary"]["Yakınsayan"]<=2).all()
    assert (a["summary"]["Yakınsama oranı"].between(0,1)).all()
    assert a["runs"]["Tohum"].nunique()==2
    assert a["runs"].groupby("Tekrar")["Tohum"].nunique().eq(1).all()


def test_repeated_recovery_rejects_invalid_inputs():
    with pytest.raises(ValueError,match="Tekrar"):
        repeated_recovery(_truth(),repetitions=0)
    with pytest.raises(ValueError,match="Örneklem"):
        repeated_recovery(_truth(),sample_sizes=(12,))
    with pytest.raises(ValueError,match="model"):
        repeated_recovery(_truth(),models=("5PL",))
