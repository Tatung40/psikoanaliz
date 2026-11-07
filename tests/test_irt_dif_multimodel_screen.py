import numpy as np
import pandas as pd
import pytest
from core.irt_dif_multimodel_screen import screen_mtk_groups

def test_binary_and_ordinal_screening():
    rng=np.random.default_rng(47)
    for model,categories in [("2PL",2),("3PL",2),("4PL",2),("GRM",4),("PCM",4),("GPCM",4),("RSM",4)]:
        x=pd.DataFrame(rng.integers(0,categories,(100,5)),columns=list("ABCDE"))
        group=pd.Series(["R"]*50+["F"]*50)
        r=screen_mtk_groups(x,list(x),group,"R","F",model,permutations=19)
        assert len(r)==5
        assert r["Yöntem"].str.contains("DIF DEĞİL").all()
        assert r["p"].dropna().between(0,1).all()

def test_nominal_labels_not_treated_as_ordered():
    rng=np.random.default_rng(18)
    x=pd.DataFrame(rng.choice(list("ABCDE"),(100,5)),columns=list("abcde"))
    result=screen_mtk_groups(x,list(x),pd.Series([0]*50+[1]*50),0,1,"NRM")
    assert len(result)==5
    assert result["Yöntem"].str.contains("DIF DEĞİL").all()

def test_small_groups_blocked():
    x=pd.DataFrame(np.random.default_rng(7).integers(0,2,(45,4)))
    with pytest.raises(ValueError,match="20"):
        screen_mtk_groups(x,list(x),pd.Series([0]*40+[1]*5),0,1,"2PL")


def test_screen_decisions_use_bh_family_and_flag_sparse_nominal_cells():
    import numpy as np
    from core.dif_classical import adjust_pvalues
    rng = np.random.default_rng(3); x = pd.DataFrame(rng.integers(0, 2, (120, 6)), columns=[f"M{i}" for i in range(6)])
    g = pd.Series(["R"] * 60 + ["F"] * 60)
    r = screen_mtk_groups(x, list(x), g, "R", "F", "2PL", permutations=19)
    assert np.allclose(r["p (BH)"], adjust_pvalues(r["p"], "BH")) and (r["p (BH)"] >= r["p"] - 1e-15).all()
    assert all((k == "Koşullu fark incelemesi önerilir") == (q < .05) for k, q in zip(r["Karar"], r["p (BH)"]))
    nom = pd.DataFrame({"A": ["x"] * 48 + ["y", "z"] * 1 + ["x"] * 50, "B": ["p", "q"] * 50, "C": ["u", "v", "w", "u"] * 25})
    n = screen_mtk_groups(nom, list(nom), pd.Series([0] * 50 + [1] * 50), 0, 1, "NRM")
    assert n.loc[n.Madde == "A", "Seyrek hücre"].item() and "seyrek hücre" in n.loc[n.Madde == "A", "Karar"].item()
