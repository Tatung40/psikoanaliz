"""
PsikoAnaliz — Uçtan Uca Bulut Simülasyon ve Fonksiyonellik Testi (tests/test_e2e_cloud_simulation.py)
Bulut üzerinde çalışacak tüm analiz akışını (veri yükleme, puanlama, madde analizi, 
çeldirici, güvenirlik, grafik üretimi ve rapor indirme) simüle eder.
Geliştirici: Doç. Dr. Mehmet Taha Eser
"""

import os
import pytest
import pandas as pd
from core.data_processing import load_dataset, handle_missing_data
from core.scoring import score_dichotomous, calculate_total_scores
from core.test_statistics import compute_test_summary, compute_frequency_distribution
from core.item_analysis import compute_item_statistics
from core.distractor_analysis import compute_all_items_distractors
from core.reliability import compute_reliability_indices
from core.group_analysis import compare_test_scores_by_group, compute_group_item_fairness
from reports.excel_export import create_excel_report
from reports.word_export import create_word_report
from ui.charts import (
    plot_score_histogram,
    plot_score_boxplot,
    plot_item_difficulty_bar,
    plot_item_discrimination_bar,
    plot_difficulty_discrimination_scatter,
    plot_alpha_if_deleted,
    plot_correlation_heatmap
)


def test_complete_e2e_pipeline_on_sample_data():
    """Tüm analiz adımlarının ardışık olarak hatasız çalıştığını doğrular."""
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")
    csv_path = os.path.join(sample_dir, "coktan_secmeli_ham.csv")
    key_path = os.path.join(sample_dir, "cevap_anahtari.csv")

    assert os.path.exists(csv_path), "Örnek çoktan seçmeli veri dosyası bulunamadı."
    assert os.path.exists(key_path), "Cevap anahtarı dosyası bulunamadı."

    # 1. Veri Okuma
    df_raw, meta = load_dataset(csv_path, file_type="csv")
    assert len(df_raw) == 350
    df_key = pd.read_csv(key_path)
    answer_key = dict(zip(df_key["Madde"], df_key["Dogru_Cevap"]))
    item_cols = [c for c in df_raw.columns if c.startswith("M_")]
    assert len(item_cols) == 30

    # 2. Puanlama (0/1)
    scored_df, diag = score_dichotomous(df_raw, item_cols, answer_key, missing_as_zero=True, id_column="Ogrenci_ID")
    assert scored_df.shape == (350, 31)

    # 3. Test İstatistikleri
    total_scores = calculate_total_scores(scored_df, item_cols)
    test_summary = compute_test_summary(total_scores, len(item_cols))
    assert test_summary["n_examinees"] == 350
    assert 0.0 <= test_summary["mean"] <= 30.0

    # 4. Madde Analizi
    item_summary, item_meta = compute_item_statistics(scored_df, item_cols, total_scores, is_dichotomous=True)
    assert len(item_summary) == 30
    assert "r_ir (Düzeltilmiş)" in item_summary.columns

    # 5. Güvenirlik Analizi (KR-20 ve Alfa Eşdeğerliği)
    rel_res = compute_reliability_indices(scored_df, item_cols, is_dichotomous=True)
    assert abs(rel_res["cronbach_alpha"] - rel_res["kr20"]) < 1e-12
    assert rel_res["sem"] > 0.0

    # 6. Çeldirici Analizi
    all_distractors = compute_all_items_distractors(df_raw, item_cols, answer_key, total_scores, scored_df=scored_df)
    assert len(all_distractors) == 30

    # 7. Grup Karşılaştırması & Adillik
    group_comp = compare_test_scores_by_group(df_raw, "Cinsiyet", total_scores)
    assert "student_t" in group_comp or "welch_t" in group_comp
    fairness_df, f_meta = compute_group_item_fairness(df_raw, "Cinsiyet", item_cols)
    assert len(fairness_df) == 30

    # 8. Plotly Grafikleri Üretimi
    fig_hist = plot_score_histogram(total_scores)
    assert fig_hist is not None
    fig_box = plot_score_boxplot(total_scores)
    assert fig_box is not None
    fig_diff = plot_item_difficulty_bar(item_summary)
    assert fig_diff is not None
    fig_disc = plot_item_discrimination_bar(item_summary)
    assert fig_disc is not None
    fig_scat = plot_difficulty_discrimination_scatter(item_summary)
    assert fig_scat is not None
    fig_alpha = plot_alpha_if_deleted(item_summary, rel_res["cronbach_alpha"])
    assert fig_alpha is not None
    corr_mat = scored_df[item_cols].corr()
    fig_heat = plot_correlation_heatmap(corr_mat)
    assert fig_heat is not None

    # 9. Excel Raporu (.xlsx) Üretimi
    freq_df = compute_frequency_distribution(total_scores)
    excel_io = create_excel_report(test_summary, item_summary, rel_res, distractor_data=all_distractors, frequency_df=freq_df)
    assert excel_io is not None
    excel_bytes = excel_io.getvalue()
    assert len(excel_bytes) > 2000

    # 10. Word Raporu (.docx) Üretimi
    word_io = create_word_report(test_summary, item_summary, rel_res, distractor_data=all_distractors)
    assert word_io is not None
    word_bytes = word_io.getvalue()
    assert len(word_bytes) > 2000
