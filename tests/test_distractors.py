"""
PsikoAnaliz — Çeldirici Analizi Birim Testleri (tests/test_distractors.py)
Seçenek frekansları, üst-alt %27 seçim oranları, D_opt ve tanısal çeldirici taraması.
Geliştirici: Doç. Dr. Mehmet Taha Eser
"""

import pytest
import numpy as np
import pandas as pd
from core.distractor_analysis import analyze_distractors_for_item


def test_distractor_analysis_proportions_and_discrimination():
    """Çeldirici oranları ve seçenek ayırt ediciliği testi."""
    # 100 öğrenci: İlk 50'si düşük puanlı (Total=10), son 50'si yüksek puanlı (Total=90)
    # Madde Doğru Cevabı: A
    # Yüksek puanlılar A seçmiş (40 kişi), B seçmiş (10 kişi)
    # Düşük puanlılar A seçmiş (10 kişi), B seçmiş (20 kişi), C seçmiş (20 kişi)
    resp = ["A"] * 10 + ["B"] * 20 + ["C"] * 20 + ["A"] * 40 + ["B"] * 10
    totals = [10.0] * 50 + [90.0] * 50

    raw_s = pd.Series(resp)
    tot_s = pd.Series(totals)

    res = analyze_distractors_for_item(
        raw_responses=raw_s,
        correct_option="A",
        total_scores=tot_s,
        non_functioning_threshold_pct=5.0
    )

    opts_df = res["options_table"]
    row_a = opts_df[opts_df["Seçenek"] == "A"].iloc[0]
    row_b = opts_df[opts_df["Seçenek"] == "B"].iloc[0]
    row_c = opts_df[opts_df["Seçenek"] == "C"].iloc[0]

    # Doğru Cevap (A) için D_opt pozitif olmalıdır (üst grup daha çok seçti)
    assert row_a["Doğru Cevap"] == "Evet"
    assert row_a["Ayırt Edicilik (D_opt)"] > 0

    # Çeldirici C için üst grup %0, alt grup %40 -> D_opt = -0.40 (Başarılı çeldirici)
    assert row_c["Doğru Cevap"] == "Hayır"
    assert row_c["Ayırt Edicilik (D_opt)"] < 0


def test_non_functioning_distractor_screening():
    """Çalışmayan (tercih edilmeyen, < %5) çeldirici tespiti testi."""
    # D seçeneğini yalnızca 1 kişi seçmiş (%1), E hiç seçilmemiş (%0)
    resp = ["A"] * 60 + ["B"] * 20 + ["C"] * 19 + ["D"] * 1
    totals = list(range(100))

    res = analyze_distractors_for_item(
        raw_responses=pd.Series(resp),
        correct_option="A",
        total_scores=pd.Series(totals),
        non_functioning_threshold_pct=5.0,
        options_list=["A", "B", "C", "D", "E"]
    )

    non_func = res["non_functioning_distractors"]
    assert "D" in non_func
    assert "E" in non_func


def test_inverted_distractor_detection():
    """Ters çalışan (üst grubu alt gruptan daha çok çeken) çeldirici tespiti testi."""
    # Çeldirici B'yi üst grubun %40'ı, alt grubun ise sadece %10'u seçmiş olsun
    resp = ["A"] * 20 + ["B"] * 5 + ["C"] * 25 + ["A"] * 20 + ["B"] * 25 + ["C"] * 5
    totals = [10.0] * 50 + [90.0] * 50

    res = analyze_distractors_for_item(
        raw_responses=pd.Series(resp),
        correct_option="A",
        total_scores=pd.Series(totals)
    )

    assert "B" in res["inverted_distractors"]
    assert "KRİTİK SORUN" in res["overall_evaluation"]
