from core.irt_engine_policy import select_engine
import pytest


def test_library_first_when_both_certified():
    d = select_engine("2PL", has_missing=False,
                      certified={"native": {"2PL": True},
                                 "python_mirt": {"2PL": True}})
    assert d.engine == "python_mirt"
    assert d.status == "certified"


def test_native_allowed_only_when_certified():
    d = select_engine("3PL", has_missing=False,
                      certified={"native": {"3PL": True}})
    assert d.engine == "native"


def test_no_unvalidated_fallback():
    d = select_engine("4PL", has_missing=False)
    assert d.engine is None and d.status == "blocked"


def test_missing_requires_separate_certification():
    d = select_engine("2PL", has_missing=True,
                      certified={"python_mirt": {"2PL": True}})
    assert d.status == "blocked"


def test_missing_certified_separately():
    d = select_engine("2PL", has_missing=True,
                      certified={"python_mirt": {"2PL:missing": True}})
    assert d.engine == "python_mirt"


def test_invalid_model():
    with pytest.raises(ValueError):
        select_engine("5PL", has_missing=False)
