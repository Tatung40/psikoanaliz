import pytest
from core.irt_engine_registry import ENGINES, engine_inventory, select_engine

def test_registry_covers_models():
    assert {"GRM","PCM","GPCM","RSM","NRM","MIRT","BIFACTOR","DIF"}.issubset(
        set(ENGINES["python_mirt"].models))
    assert "MIXED_DICHOTOMOUS" in ENGINES["eser"].models

def test_inventory_does_not_claim_scientific_validation():
    rows=engine_inventory()
    assert len(rows)==4
    assert all(not x["production_ready"] and x["validated_models"]==[] for x in rows)
    assert all(isinstance(x["installed"],bool) for x in rows)

@pytest.mark.parametrize("model,engine",[("GRM","python_mirt"),("2PL","eser"),("4PL","py_irt")])
def test_experimental_engines_fail_closed(model,engine):
    with pytest.raises(RuntimeError,match="not production-authorized"):
        select_engine(model,engine)

def test_unknown_engine_and_model_rejected():
    with pytest.raises(ValueError,match="Unknown engine"):
        select_engine("2PL","nonexistent")
    with pytest.raises(ValueError,match="No registered"):
        select_engine("UNKNOWN")
