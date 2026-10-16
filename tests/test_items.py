"""
PsikoAnaliz — Madde ve Test Analizi Birim Testleri (tests/test_items.py)
Madde güçlüğü, ayırt ediciliği, r_it, r_ir, r_pbis, r_bis, D_27 ve test düzey istatistikleri.
Geliştirici: Doç. Dr. Mehmet Taha Eser
"""

import pytest
import numpy as np
import pandas as pd
from core.item_analysis import compute_item_statistics
from core.test_statistics import compute_test_summary, compute_standard_scores, compute_frequency_distribution


def test_item_difficulty_and_variance():
    """Madde güçlük indeksi (p) ve varyans doğrulaması."""
    # 4 katılımcı, 1 madde
    # Puanlar: 1, 1, 1, 0 -> p = 3/4 = 0.75, q = 0.25
    df = pd.DataFrame({"M1": [1, 1, 1, 0], "M2": [1, 0, 1, 0]})
    summary_df, _ = compute_item_statistics(df, ["M1", "M2"], is_dichotomous=True)

    m1_row = summary_df[summary_df["Madde"] == "M1"].iloc[0]
    assert m1_row["Güçlük (p)"] == 0.75
    assert m1_row["q (1-p)"] == 0.25
    # Örneklem varyansı: s^2 = (N/(N-1)) * p * q = (4/3) * 0.75 * 0.25 = 0.25
    assert pytest.approx(m1_row["Varyans"], rel=1e-3) == 0.25


def test_corrected_item_total_correlation():
    """Düzeltilmiş madde-toplam korelasyonunun (r_ir / madde-kalan) doğrulanması."""
    rng = np.random.default_rng(101)
    # 100 katılımcı, 5 madde
    data = rng.binomial(1, 0.5, size=(100, 5))
    cols = [f"M_{i+1}" for i in range(5)]
    df = pd.DataFrame(data, columns=cols)

    summary_df, _ = compute_item_statistics(df, cols, is_dichotomous=True)

    for _, row in summary_df.iterrows():
        # r_ir her zaman r_it'den küçük veya eşittir çünkü maddenin kendisiyle yapay şişirilmesi çıkarılmıştır
        if not np.isnan(row["r_it"]) and not np.isnan(row["r_ir (Düzeltilmiş)"]):
            assert row["r_ir (Düzeltilmiş)"] <= row["r_it"] + 1e-4


def test_biserial_and_point_biserial_relationship():
    """Çift-serili (r_bis) ve nokta-çift serili (r_pbis) korelasyon bağıntısının doğrulanması."""
    rng = np.random.default_rng(202)
    # Orta güçlükte madde üret (p ~ 0.5)
    data = rng.binomial(1, 0.5, size=(200, 10))
    cols = [f"I_{i}" for i in range(10)]
    df = pd.DataFrame(data, columns=cols)

    summary_df, _ = compute_item_statistics(df, cols, is_dichotomous=True)

    for _, row in summary_df.iterrows():
        p = row["Güçlük (p)"]
        r_pbis = row["r_pbis"]
        r_bis = row["r_bis"]
        # p ~ 0.5 civarında r_bis / r_pbis oranı yaklaşık 1.253'tür (sqrt(pi/2))
        if 0.35 <= p <= 0.65 and not np.isnan(r_pbis) and not np.isnan(r_bis) and r_pbis > 0.1:
            ratio = r_bis / r_pbis
            assert 1.10 <= ratio <= 1.45


def test_kelley_upper_lower_discrimination_with_ties():
    """Üst-Alt %27 grup ayırt ediciliği ve bağ (tie) yönetiminin doğrulanması."""
    # Belirgin şekilde ayırt eden madde
    # Yüksek puanlılar 1, düşük puanlılar 0
    scores = np.arange(1, 101)  # 1'den 100'e puanlar
    # Madde: Puanı 50'den büyük olanlar 1, küçük olanlar 0
    item_val = (scores > 50).astype(int)
    other_items = np.tile([0, 1], 50)
    df = pd.DataFrame({"Target": item_val, "Dummy": other_items})

    summary_df, meta = compute_item_statistics(df, ["Target", "Dummy"], total_scores=pd.Series(scores))
    target_row = summary_df[summary_df["Madde"] == "Target"].iloc[0]

    # Üst %27 grupta güçlük 1.0, alt %27 grupta 0.0 -> D = 1.0
    assert target_row["Üst Grup p"] == 1.0
    assert target_row["Alt Grup p"] == 0.0
    assert target_row["D (%27)"] == 1.0


def test_test_level_statistics():
    """Test düzeyi betimsel istatistikler, standart puanlar ve frekans dağılımı testi."""
    scores = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    summary = compute_test_summary(scores, k_items=10, theoretical_min_score=0, theoretical_max_score=100)

    assert summary["n_examinees"] == 10
    assert summary["mean"] == 55.0
    assert summary["min"] == 10.0
    assert summary["max"] == 100.0
    assert summary["range"] == 90.0

    # z ve T puanları
    std_df = compute_standard_scores(scores)
    assert pytest.approx(std_df["z_score"].mean(), abs=1e-5) == 0.0
    assert pytest.approx(std_df["t_score"].mean(), abs=1e-5) == 50.0

    # Frekans tablosu
    freq_df = compute_frequency_distribution(scores)
    assert len(freq_df) == 10
    assert freq_df["Kümülatif Yüzde (%)"].iloc[-1] == 100.0
