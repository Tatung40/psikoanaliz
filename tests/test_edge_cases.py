"""
PsikoAnaliz — Uç Durumlar ve Hata Güvenliği Testleri (tests/test_edge_cases.py)
Sabit maddeler (varyans=0), negatif ayırt edicilik, sıfır toplam varyans ve küçük örneklemler.
Geliştirici: Doç. Dr. Mehmet Taha Eser
"""

import pytest
import numpy as np
import pandas as pd
from core.item_analysis import compute_item_statistics
from core.reliability import compute_reliability_indices
from core.validation import validate_analysis_inputs


def test_constant_item_zero_variance():
    """Tüm katılımcıların aynı puanı aldığı sabit madde testi (Varyans = 0)."""
    # M1 sabit (herkes 1), M2 normal
    df = pd.DataFrame({
        "M1": [1, 1, 1, 1, 1],
        "M2": [1, 0, 1, 0, 1]
    })
    summary_df, meta = compute_item_statistics(df, ["M1", "M2"], is_dichotomous=True)

    m1_row = summary_df[summary_df["Madde"] == "M1"].iloc[0]
    assert m1_row["Varyans"] == 0.0
    assert np.isnan(m1_row["r_ir (Düzeltilmiş)"])
    assert "Sabit Madde (Varyans=0)" in m1_row["Tanısal Bayrak"]


def test_zero_total_score_variance():
    """Herkesin aynı toplam puanı alması durumunda çökme yaşanmaması testi."""
    df = pd.DataFrame({
        "M1": [1, 0, 1, 0],
        "M2": [0, 1, 0, 1]  # Herkesin toplam puanı tam olarak 1! (Total variance = 0)
    })
    summary_df, meta = compute_item_statistics(df, ["M1", "M2"], is_dichotomous=True)
    rel_res = compute_reliability_indices(df, ["M1", "M2"], is_dichotomous=True)

    # Uygulama sıfıra bölme hatası vermemeli, alfa güvenli şekilde 0 dönmeli
    assert rel_res["cronbach_alpha"] == 0.0


def test_negative_discrimination_item():
    """Negatif ayırt ediciliğe sahip hatalı maddenin doğru bayraklandırılması testi."""
    # M1'i yüksek yetenekliler yanlış, düşük yetenekliler doğru yapmış olsun
    # Katılımcıların gerçek yeteneği:
    totals = np.array([10, 20, 30, 80, 90, 100])
    item_val = np.array([1, 1, 1, 0, 0, 0])  # Ters çalışan madde!
    df = pd.DataFrame({"NegItem": item_val, "Dummy": [1, 0, 1, 1, 1, 1]})

    summary_df, _ = compute_item_statistics(df, ["NegItem"], total_scores=pd.Series(totals))
    neg_row = summary_df.iloc[0]

    assert neg_row["r_it"] < 0
    assert neg_row["D (%27)"] < 0
    assert "Negatif Ayırt Edicilik (Acil İnceleme!)" in neg_row["Tanısal Bayrak"]


def test_small_sample_size_handling():
    """Çok küçük örneklemlerde (N = 3) doğrulama ve hata üretmeme testi."""
    df = pd.DataFrame({
        "M1": [1, 0, 1],
        "M2": [1, 1, 0]
    })
    is_valid, errors, warnings = validate_analysis_inputs(df, ["M1", "M2"])
    assert is_valid is True
    assert any("Katılımcı sayısı çok düşük" in w for w in warnings)
