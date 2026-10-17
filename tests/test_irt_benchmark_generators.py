import numpy as np
from scripts.irt_benchmark import reference_items

def test_generating_models_use_correct_upper_asymptote():
    m3=reference_items(25,"3PL")
    m4=reference_items(25,"4PL")
    assert np.all(m3["d"].to_numpy()==1.)
    assert np.all((m4["d"].to_numpy()<1.)&(m4["d"].to_numpy()>.65))
    assert np.allclose(m3["a"],m4["a"])
    assert np.allclose(m3["b"],m4["b"])
    assert np.allclose(m3["c"],m4["c"])
