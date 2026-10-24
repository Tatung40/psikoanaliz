import numpy as np
import pandas as pd
import pytest
from core.irt_library_parameter_mapping import (
    normalize_binary_item_parameters, curves_from_normalized_table,
)


@pytest.mark.parametrize("model,source", [
    ("1PL", {"b": [0., 1.]}),
    ("2PL", {"a": [1., 1.5], "b": [0., 1.]}),
    ("3PL", {"a": [1., 1.5], "b": [0., 1.], "c": [.1, .2]}),
    ("4PL", {"a": [1., 1.5], "b": [0., 1.], "c": [.1, .2], "d": [.9, .95]}),
])
def test_explicit_model_normalization_and_curves(model, source):
    result = normalize_binary_item_parameters(
        pd.DataFrame(source), model=model, item_names=["Q1", "Q2"]
    )
    assert list(result.columns) == ["Madde", "a", "b", "c", "d"]
    curves = curves_from_normalized_table(result, np.linspace(-3, 3, 21))
    assert curves["probability"].shape == (21, 2)
    assert np.all(curves["test_information"] >= 0)


def test_no_implicit_intercept_to_difficulty_conversion():
    with pytest.raises(ValueError, match="logistic-metric"):
        normalize_binary_item_parameters(
            pd.DataFrame({"a": [1.], "intercept": [-.2]}), model="2PL"
        )


def test_invalid_item_names_fail():
    with pytest.raises(ValueError, match="unique"):
        normalize_binary_item_parameters(
            pd.DataFrame({"a": [1., 1.], "b": [0., 0.]}),
            model="2PL", item_names=["X", "X"]
        )
