"""
PsikoAnaliz — Grup Bazlı Analizler ve Karşılaştırma Testleri (tests/test_group_analysis.py)
t-testi, Welch t, Cohen's d, ANOVA ve KTK madde adillik Delta p taraması.
Geliştirici: Doç. Dr. Mehmet Taha Eser
"""

import pytest
import numpy as np
import pandas as pd
from core.group_analysis import compare_test_scores_by_group, compute_group_item_fairness


def test_two_groups_comparison_and_cohens_d():
    """İki grup karşılaştırması (t-testi, Welch t ve Cohen's d) testi."""
    rng = np.random.default_rng(42)
    # Grup 1: Ort = 70, SS = 10, N = 50
    # Grup 2: Ort = 60, SS = 10, N = 50
    # Beklenen d ~ (70 - 60) / 10 = 1.0 (Büyük etki)
    s1 = rng.normal(70, 10, 50)
    s2 = rng.normal(60, 10, 50)

    df = pd.DataFrame({
        "Grup": ["G1"] * 50 + ["G2"] * 50,
        "Puan": np.concatenate([s1, s2])
    })

    # Beklenen d hesabı:
    s_pooled = np.sqrt(((50 - 1) * s1.var(ddof=1) + (50 - 1) * s2.var(ddof=1)) / (50 + 50 - 2))
    expected_d = (s1.mean() - s2.mean()) / s_pooled

    res = compare_test_scores_by_group(df, "Grup", df["Puan"])
    assert res["comparison_type"] == "2_groups_ttest"
    assert res["student_p"] < 0.001
    assert pytest.approx(res["cohens_d"], rel=1e-3) == expected_d
    assert res["cohens_d"] >= 0.80
    assert "Geniş / Büyük etki" in res["effect_size_interpretation"]


def test_three_groups_anova():
    """Üç veya daha fazla grup için Tek Yönlü ANOVA ve eta-kare testi."""
    rng = np.random.default_rng(88)
    s1 = rng.normal(50, 5, 40)
    s2 = rng.normal(55, 5, 40)
    s3 = rng.normal(65, 5, 40)

    df = pd.DataFrame({
        "Sube": ["A"] * 40 + ["B"] * 40 + ["C"] * 40,
        "Puan": np.concatenate([s1, s2, s3])
    })

    res = compare_test_scores_by_group(df, "Sube", df["Puan"])

    assert res["comparison_type"] == "multi_group_anova"
    assert res["anova_p"] < 0.001
    assert 0.20 <= res["eta_squared"] <= 0.80


def test_group_item_fairness_delta_p():
    """KTK madde güçlük farkı (Delta p) ön tarama testi."""
    # M1'de Grup 1 p = 0.90, Grup 2 p = 0.50 -> Delta p = 0.40 (Belirgin fark)
    # M2'de Grup 1 p = 0.60, Grup 2 p = 0.60 -> Delta p = 0.00 (Dengeli)
    df = pd.DataFrame({
        "Grup": ["A"] * 50 + ["B"] * 50,
        "M1": [1] * 45 + [0] * 5 + [1] * 25 + [0] * 25,
        "M2": [1] * 30 + [0] * 20 + [1] * 30 + [0] * 20
    })

    fairness_df, meta = compute_group_item_fairness(df, "Grup", ["M1", "M2"], delta_p_threshold=0.15)

    m1_row = fairness_df[fairness_df["Madde"] == "M1"].iloc[0]
    m2_row = fairness_df[fairness_df["Madde"] == "M2"].iloc[0]

    assert pytest.approx(m1_row["Fark (Δp = p_ref - p_foc)"], abs=1e-3) == 0.40
    assert "Önemli Güçlük Farkı" in m1_row["Ön Tarama Durumu"]
    assert "Dengeli (Fark yok)" in m2_row["Ön Tarama Durumu"]
