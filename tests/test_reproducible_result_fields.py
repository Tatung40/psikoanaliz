"""Python code panel keeps the actual producer when a later page run uses a field of a stored result
(e.g. PythonMirtResult.fit); the generated recipe replays the real library calls."""
import dataclasses
from pathlib import Path

import importlib
import inspect
import numpy as np
import pandas as pd
import pytest

from core.reproducible import begin, decode, finish, python_code, recorded


def _replay(manifest):
    results = []
    for call in manifest["calls"]:
        function = importlib.import_module(call["module"])
        for part in call["function"].split("."):
            function = getattr(function, part)
        bound = inspect.signature(function).bind_partial()
        bound.arguments.update(decode(call["arguments"], results))
        results.append(function(*bound.args, **bound.kwargs))
    return results


def test_field_of_cached_dataclass_result_imports_its_producer_without_refit():
    @dataclasses.dataclass
    class Outcome:
        model: str
        fit: object
    calls = []
    @recorded
    def fit(x, setting):
        calls.append(setting)
        return Outcome("2PL", {"loglik": float(x.to_numpy().sum())})
    @recorded
    def diagnostics(library_fit):
        return pd.DataFrame({"loglik": [library_fit["loglik"]]})
    first = begin("model")
    outcome = fit(pd.DataFrame({"a": [1., 2.]}), setting="original")
    finish()
    second = begin("model", previous=first)
    diagnostics(outcome.fit)
    finish()
    assert calls == ["original"]
    assert [c["function"] for c in second.calls] == ["fit", "diagnostics"]
    ref = second.calls[1]["arguments"]["values"][0][1]
    assert ref == {"@": "ref", "path": [0, {"@attr": "fit"}]}
    assert decode(ref, [outcome]) is outcome.fit


@pytest.mark.parametrize("model", ["2PL", "3PL"])
def test_binary_irt_page_code_survives_rerun_and_replays_the_real_fit(model):
    pytest.importorskip("mirt")
    from streamlit.testing.v1 import AppTest
    from core import example_datasets as ex
    case = {"2PL": "irt_006", "3PL": "irt_009"}[model]
    b = ex.materialize(case); x = b["data"][b["items"]].astype(int)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=300)
    for key, value in dict(raw_df=x, scored_df=x.copy(), item_cols=list(x.columns), id_col=None, group_col=None,
                           is_dichotomous=True, nav_page=f"MTK: {model}").items():
        at.session_state[key] = value
    at.run()
    at.button(key=f"py_mirt_run_{model}").click().run()
    at.run()                                         # an ordinary rerun (widget change, tab switch)
    assert not at.exception
    assert not any("UNSUPPORTED" in i.value for i in at.info)
    manifest = at.session_state["_repro_history"].manifest()
    functions = [c["function"] for c in manifest["calls"]]
    assert manifest["status"] == "GENERATED / UNVERIFIED" and functions[0] == "fit_python_mirt"
    assert {"library_fit_diagnostics", "python_mirt_binary_parameters"} <= set(functions)
    compile(python_code(manifest), "analysis.py", "exec")
    replayed = _replay(manifest)
    stored = at.session_state["python_mirt_experimental_results"][model]["fit"].fit
    np.testing.assert_allclose(replayed[0].fit.log_likelihood, stored.log_likelihood, rtol=0, atol=1e-8)
    from core.irt_python_mirt_parameters import python_mirt_binary_parameters
    expected = python_mirt_binary_parameters(stored, model=model, item_names=list(x.columns))
    pd.testing.assert_frame_equal(replayed[functions.index("python_mirt_binary_parameters")], expected,
                                  check_exact=False, atol=1e-8, rtol=0)
