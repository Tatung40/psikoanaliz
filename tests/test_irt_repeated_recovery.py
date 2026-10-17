import numpy as np
import pandas as pd
import pytest
from core.irt_simulation import repeated_recovery


def _truth(k=5):
    # k=5 keeps both 3PL (15 <= 31) and 4PL (20 <= 31) structurally identifiable.
    return pd.DataFrame({
        "Madde":["M1","M2","M3","M4","M5"],
        "a":[.8,1.2,1.5,1.,1.3],"b":[-.6,0.,.7,-.2,.4],
        "c":[.08,.12,.14,.1,.11],"d":[.92,.94,.96,.93,.95]}).head(k)


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
    assert not a["runs"]["Durum"].eq("Hata").any()   # real estimation path exercised


def test_unidentified_models_are_logged_as_failures():
    out=repeated_recovery(_truth(3),models=("3PL","4PL"),sample_sizes=(80,),
                          repetitions=1,seed=81,points=11,max_iter=2)
    assert out["runs"]["Durum"].eq("Hata").all()
    assert out["runs"]["Hata"].str.contains("yapısal olarak tanımlanamaz").all()
    assert (out["summary"]["Yakınsayan"]==0).all()


def test_repeated_recovery_rejects_invalid_inputs():
    with pytest.raises(ValueError,match="Tekrar"):
        repeated_recovery(_truth(),repetitions=0)
    with pytest.raises(ValueError,match="Örneklem"):
        repeated_recovery(_truth(),sample_sizes=(12,))
    with pytest.raises(ValueError,match="model"):
        repeated_recovery(_truth(),models=("5PL",))
