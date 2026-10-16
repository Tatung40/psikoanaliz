"""
PsikoAnaliz — Puanlama ve Ölçekleme Birim Testleri (tests/test_scoring.py)
Geliştirici: Doç. Dr. Mehmet Taha Eser
"""

import pytest
import numpy as np
import pandas as pd
from core.scoring import (
    score_dichotomous,
    reverse_score_items,
    validate_polytomous_data,
    calculate_total_scores
)


def test_dichotomous_scoring_basic():
    """Temel çoktan seçmeli cevap anahtarı puanlama testi."""
    raw_data = pd.DataFrame({
        "Ogrenci": ["Ali", "Ayse", "Fatma"],
        "M1": ["A", "B", "A"],
        "M2": ["C", "C", "D"],
        "M3": ["B", "B", "B"]
    })
    key = {"M1": "A", "M2": "C", "M3": "B"}
    scored_df, diag = score_dichotomous(
        raw_data,
        item_columns=["M1", "M2", "M3"],
        answer_key=key,
        case_sensitive=False
    )

    # Ali: A, C, B -> 1, 1, 1
    # Ayse: B, C, B -> 0, 1, 1
    # Fatma: A, D, B -> 1, 0, 1
    assert scored_df.loc[0, "M1"] == 1.0
    assert scored_df.loc[0, "M2"] == 1.0
    assert scored_df.loc[0, "M3"] == 1.0

    assert scored_df.loc[1, "M1"] == 0.0
    assert scored_df.loc[1, "M2"] == 1.0
    assert scored_df.loc[1, "M3"] == 1.0

    assert scored_df.loc[2, "M1"] == 1.0
    assert scored_df.loc[2, "M2"] == 0.0
    assert scored_df.loc[2, "M3"] == 1.0


def test_dichotomous_case_insensitivity_and_whitespace():
    """Büyük-küçük harf ve boşluk duyarsızlığı testi."""
    raw_data = pd.DataFrame({
        "M1": [" a ", "A", "b", "B "]
    })
    key = {"M1": "A"}
    scored_df, _ = score_dichotomous(raw_data, ["M1"], key, case_sensitive=False)
    assert list(scored_df["M1"]) == [1.0, 1.0, 0.0, 0.0]


def test_dichotomous_missing_handling():
    """Boş bırakılan yanıtların NaN vs 0 olarak puanlanması testi."""
    raw_data = pd.DataFrame({
        "M1": ["A", np.nan, "", "B"]
    })
    key = {"M1": "A"}

    # 1. NaN olarak bırakma
    scored_nan, _ = score_dichotomous(raw_data, ["M1"], key, missing_as_zero=False)
    assert scored_nan.loc[0, "M1"] == 1.0
    assert np.isnan(scored_nan.loc[1, "M1"])
    assert np.isnan(scored_nan.loc[2, "M1"])
    assert scored_nan.loc[3, "M1"] == 0.0

    # 2. Sıfır olarak puanlama
    scored_zero, _ = score_dichotomous(raw_data, ["M1"], key, missing_as_zero=True)
    assert scored_zero.loc[0, "M1"] == 1.0
    assert scored_zero.loc[1, "M1"] == 0.0
    assert scored_zero.loc[2, "M1"] == 0.0
    assert scored_zero.loc[3, "M1"] == 0.0


def test_reverse_scoring_theoretical_bounds():
    """
    Ters puanlamanın kuramsal ölçek sınırlarına dayalı olarak (Min + Max - X)
    doğru hesaplandığının ve ampirik min/max hatasına düşülmediğinin doğrulanması.
    """
    # 1-5 Likert ölçeğinde katılımcılar sadece 2, 3, 4 seçmiş olsun (ampirik min=2, max=4)
    # Eğer yanlışlıkla ampirik min/max kullanılsaydı 2+4-X = 6-X yerine başka çıkardı.
    df = pd.DataFrame({
        "M1_Ters": [2, 3, 4, 2]
    })
    # Kuramsal sınırlar: min=1, max=5 -> Toplam=6
    rev_df, report = reverse_score_items(
        df,
        reverse_items=["M1_Ters"],
        theoretical_min=1.0,
        theoretical_max=5.0
    )
    # 2 -> 6 - 2 = 4
    # 3 -> 6 - 3 = 3
    # 4 -> 6 - 4 = 2
    expected = [4.0, 3.0, 2.0, 4.0]
    assert list(rev_df["M1_Ters"]) == expected
    assert len(report["out_of_bounds_warnings"]) == 0


def test_reverse_scoring_out_of_bounds_detection():
    """Kuramsal sınırların dışındaki hatalı girişlerin tespit edilmesi testi."""
    df = pd.DataFrame({
        "M1_Ters": [1, 5, 6, 0]  # 6 ve 0 geçersiz (1-5 aralığı dışında)
    })
    _, report = reverse_score_items(
        df,
        reverse_items=["M1_Ters"],
        theoretical_min=1.0,
        theoretical_max=5.0
    )
    assert "M1_Ters" in report["out_of_bounds_warnings"]
    assert report["out_of_bounds_warnings"]["M1_Ters"]["count"] == 2


def test_total_score_calculation():
    """Farklı kayıp veri kuralları altında toplam puan hesaplama testi."""
    df = pd.DataFrame({
        "M1": [1.0, 1.0, np.nan],
        "M2": [1.0, np.nan, np.nan],
        "M3": [1.0, 1.0, 1.0]
    })

    # strict kuralı
    tot_strict = calculate_total_scores(df, ["M1", "M2", "M3"], missing_rule="strict")
    assert tot_strict[0] == 3.0
    assert np.isnan(tot_strict[1])
    assert np.isnan(tot_strict[2])

    # zero_impute kuralı
    tot_zero = calculate_total_scores(df, ["M1", "M2", "M3"], missing_rule="zero_impute")
    assert tot_zero[0] == 3.0
    assert tot_zero[1] == 2.0
    assert tot_zero[2] == 1.0
