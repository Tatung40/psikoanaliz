"""Python library UI cache must fail closed when data or calculations change."""
import ast
import pandas as pd
from pathlib import Path
from ui.irt_python_library_view import _data_fingerprint

def test_fingerprint_changes_with_values_order_columns_and_index():
    x=pd.DataFrame({"A":[0,1,0],"B":[1,0,1]})
    original=_data_fingerprint(x)
    assert original==_data_fingerprint(x.copy())
    assert original!=_data_fingerprint(x.iloc[::-1])
    assert original!=_data_fingerprint(x[["B","A"]])
    changed=x.copy()
    changed.loc[0,"A"]=1
    assert original!=_data_fingerprint(changed)
    assert original!=_data_fingerprint(x.reset_index(drop=True).set_axis([5,6,7]))

def test_stale_python_mirt_result_is_cleared_at_start_and_on_change():
    source=Path("ui/irt_python_library_view.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert 'st.session_state.pop(f"python_mirt_fit_{model}", None)' in source
    assert source.count('st.session_state.pop(f"python_mirt_fit_{model}", None)') >= 3
