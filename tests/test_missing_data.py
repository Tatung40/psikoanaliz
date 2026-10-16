"""
PsikoAnaliz — Kayıp Veri Yönetimi Testleri (tests/test_missing_data.py)
Listwise, sıfır atama ve ortalama ikamesi protokollerinin doğrulanması.
Geliştirici: Doç. Dr. Mehmet Taha Eser
"""

import pytest
import numpy as np
import pandas as pd
from core.data_processing import handle_missing_data


def test_missing_data_listwise_deletion():
    """Listwise silme yönteminin doğrulanması."""
    df = pd.DataFrame({
        "M1": [1.0, 2.0, np.nan, 4.0],
        "M2": [5.0, np.nan, 7.0, 8.0]
    })
    # Sadece 0. ve 3. satırlar eksiksiz
    cleaned_df, info = handle_missing_data(df, method="listwise", item_columns=["M1", "M2"])

    assert len(cleaned_df) == 2
    assert info["dropped_rows"] == 2
    assert info["final_rows"] == 2
    assert list(cleaned_df.index) == [0, 3]


def test_missing_data_zero_imputation():
    """Eksik yanıtlara 0 atanması yönteminin doğrulanması."""
    df = pd.DataFrame({
        "M1": [1.0, np.nan, 0.0],
        "M2": [np.nan, 1.0, np.nan]
    })
    imputed_df, info = handle_missing_data(df, method="zero", item_columns=["M1", "M2"])

    assert imputed_df.isna().sum().sum() == 0
    assert info["imputed_cells"] == 3
    assert imputed_df.loc[1, "M1"] == 0.0
    assert imputed_df.loc[0, "M2"] == 0.0
    assert imputed_df.loc[2, "M2"] == 0.0


def test_missing_data_mean_imputation():
    """Madde ortalaması ile doldurma yönteminin doğrulanması."""
    df = pd.DataFrame({
        "M1": [2.0, 4.0, np.nan]  # Ortalaması = (2 + 4) / 2 = 3.0
    })
    imputed_df, info = handle_missing_data(df, method="mean", item_columns=["M1"])

    assert imputed_df.loc[2, "M1"] == 3.0
    assert info["imputed_cells"] == 1
