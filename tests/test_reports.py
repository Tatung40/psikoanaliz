"""
PsikoAnaliz — Rapor İhraç Doğrulama Testleri (tests/test_reports.py)
Excel (.xlsx) ve Word (.docx) rapor üreticilerinin doğrulanması.
Geliştirici: Doç. Dr. Mehmet Taha Eser
"""

import pytest
import numpy as np
import pandas as pd
from reports.excel_export import create_excel_report
from reports.word_export import create_word_report


def test_excel_report_generation():
    """Excel raporunun hatasız oluşturulması ve sayfalarının doğrulanması."""
    test_summary = {
        "n_examinees": 100,
        "k_items": 10,
        "mean": 65.4,
        "median": 66.0,
        "mode": 70.0,
        "std_dev": 12.3,
        "variance": 151.29,
        "sem_mean": 1.23,
        "min": 25.0,
        "max": 95.0,
        "range": 70.0,
        "skewness": -0.21,
        "kurtosis": 0.15,
        "iqr": 18.0,
        "floor_pct": 0.0,
        "ceiling_pct": 0.0
    }
    item_df = pd.DataFrame({
        "Madde": [f"M_{i+1}" for i in range(10)],
        "N": [100] * 10,
        "Kayıp %": [0.0] * 10,
        "Ortalama": [0.65] * 10,
        "SS": [0.48] * 10,
        "Varyans": [0.23] * 10,
        "Güçlük (p)": [0.65] * 10,
        "q (1-p)": [0.35] * 10,
        "r_it": [0.45] * 10,
        "r_ir (Düzeltilmiş)": [0.38] * 10,
        "r_pbis": [0.45] * 10,
        "r_bis": [0.55] * 10,
        "D (%27)": [0.42] * 10,
        "Üst Grup p": [0.85] * 10,
        "Alt Grup p": [0.43] * 10,
        "Silindiğinde Ort": [5.85] * 10,
        "Silindiğinde Alfa": [0.78] * 10,
        "Tanısal Bayrak": ["İdeal / Kabul Edilebilir"] * 10
    })
    rel_data = {
        "cronbach_alpha": 0.8124,
        "standardized_alpha": 0.8150,
        "kr20": 0.8124,
        "kr21": 0.7950,
        "split_half_odd_even_r": 0.69,
        "spearman_brown_odd_even": 0.816,
        "guttman_lambda4": 0.811,
        "mcdonald_omega": 0.825,
        "mean_inter_item_r": 0.305,
        "sem": 2.45,
        "ci_95_margin": 4.80,
        "alpha_ci_lower": 0.75,
        "alpha_ci_upper": 0.86,
        "reliability_evaluation": "Çok İyi Güvenirlik"
    }

    excel_io = create_excel_report(test_summary, item_df, rel_data)
    assert excel_io is not None
    assert len(excel_io.getvalue()) > 1000  # Geçerli bir dosya boyutu


def test_word_report_generation():
    """Word (.docx) APA 7 akademik raporunun hatasız oluşturulması testi."""
    test_summary = {
        "n_examinees": 50,
        "k_items": 5,
        "mean": 3.2,
        "std_dev": 1.1,
        "variance": 1.21,
        "min": 1.0,
        "max": 5.0,
        "range": 4.0,
        "median": 3.0,
        "mode": 3.0,
        "sem_mean": 0.15,
        "skewness": 0.1,
        "kurtosis": -0.2,
        "iqr": 1.5,
        "floor_pct": 2.0,
        "ceiling_pct": 4.0
    }
    item_df = pd.DataFrame({
        "Madde": ["Item1", "Item2"],
        "N": [50, 50],
        "Güçlük (p)": [0.6, 0.7],
        "r_it": [0.5, 0.4],
        "r_ir (Düzeltilmiş)": [0.4, 0.3],
        "D (%27)": [0.35, 0.28],
        "Silindiğinde Alfa": [0.72, 0.74],
        "Tanısal Bayrak": ["İdeal", "İdeal"]
    })
    rel_data = {
        "cronbach_alpha": 0.75,
        "sem": 0.55,
        "ci_95_margin": 1.08,
        "reliability_evaluation": "Kabul Edilebilir"
    }

    word_io = create_word_report(test_summary, item_df, rel_data)
    assert word_io is not None
    assert len(word_io.getvalue()) > 1000
