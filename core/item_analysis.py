"""
PsikoAnaliz — Klasik Madde Analizi Modülü (core/item_analysis.py)
Madde güçlüğü, ayırt ediciliği, r_it, r_ir, r_pbis, r_bis, D_%27 ve tanısal bayraklar.
Geliştirici: Doç. Dr. Mehmet Taha Eser
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from scipy import stats


def compute_item_statistics(
    df: pd.DataFrame,
    item_columns: List[str],
    total_scores: Optional[pd.Series] = None,
    is_dichotomous: bool = True,
    theoretical_min_item: float = 0.0,
    theoretical_max_item: float = 1.0,
    overall_alpha: Optional[float] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Test maddeleri için kapsamlı klasik madde istatistiklerini hesaplar.
    
    Parametreler:
    - df: Katılımcıların madde puanlarını içeren matris
    - item_columns: Madde sütun adları
    - total_scores: Katılımcı toplam puan serisi (Yoksa maddelerin toplamından hesaplanır)
    - is_dichotomous: Maddelerin ikili (0/1) olup olmadığı
    - theoretical_min_item: Maddelerin kuramsal minimum puanı (Varsayılan: 0.0)
    - theoretical_max_item: Maddelerin kuramsal maksimum puanı (Varsayılan: 1.0)
    - overall_alpha: Testin genel Cronbach alfa katsayısı (Madde silindiğinde karşılaştırma için)
    
    Dönüş: (madde_istatistikleri_df, ozet_meta)
    """
    # Sayısal matris ve temizleme
    item_df = df[item_columns].apply(pd.to_numeric, errors="coerce")
    n_examinees, k_items = item_df.shape

    if total_scores is None:
        total_scores = item_df.sum(axis=1, skipna=True)
    else:
        total_scores = total_scores.copy()

    # Üst ve Alt %27 Gruplarının Belirlenmesi (Kelley, 1939 Kuralı)
    # Bağ (tie) politikası: Yüzdelik kesme değerlerine göre eşit puanları dahil etme
    clean_total = total_scores.dropna()
    p27_cutoff = float(np.percentile(clean_total, 27))
    p73_cutoff = float(np.percentile(clean_total, 73))

    lower_group_mask = total_scores <= p27_cutoff
    upper_group_mask = total_scores >= p73_cutoff

    n_lower = int(lower_group_mask.sum())
    n_upper = int(upper_group_mask.sum())

    total_var = float(total_scores.var(ddof=1)) if n_examinees > 1 else 0.0
    total_sd = float(total_scores.std(ddof=1)) if n_examinees > 1 else 0.0
    total_mean = float(total_scores.mean())

    item_results = []
    item_range = theoretical_max_item - theoretical_min_item
    if item_range <= 0:
        item_range = 1.0

    for col in item_columns:
        col_series = item_df[col]
        valid_series = col_series.dropna()
        n_valid = len(valid_series)
        n_missing = n_examinees - n_valid
        missing_pct = (n_missing / n_examinees * 100.0) if n_examinees > 0 else 0.0

        if n_valid == 0:
            item_results.append({
                "Madde": col,
                "N": 0,
                "Kayıp %": round(missing_pct, 2),
                "Ortalama": np.nan,
                "SS": np.nan,
                "Varyans": np.nan,
                "Güçlük (p)": np.nan,
                "q (1-p)": np.nan,
                "r_it": np.nan,
                "r_ir (Düzeltilmiş)": np.nan,
                "r_pbis": np.nan,
                "r_bis": np.nan,
                "D (%27)": np.nan,
                "Üst Grup p": np.nan,
                "Alt Grup p": np.nan,
                "Silindiğinde Ort": np.nan,
                "Silindiğinde Varyans": np.nan,
                "Silindiğinde Alfa": np.nan,
                "Tanısal Bayrak": "Tüm Veri Kayıp"
            })
            continue

        item_mean = float(valid_series.mean())
        item_median = float(valid_series.median())
        item_var = float(valid_series.var(ddof=1)) if n_valid > 1 else 0.0
        item_sd = float(valid_series.std(ddof=1)) if n_valid > 1 else 0.0
        item_min = float(valid_series.min())
        item_max = float(valid_series.max())

        # Madde Güçlüğü (p)
        if is_dichotomous:
            p_val = item_mean
            q_val = 1.0 - p_val
        else:
            # Çok kategorili normalize güçlük
            p_val = (item_mean - theoretical_min_item) / item_range
            q_val = 1.0 - p_val

        # Kalan Puan (Rest Score = Total - Item)
        rest_score = total_scores - col_series.fillna(0.0)
        rest_valid_mask = col_series.notna() & total_scores.notna()
        valid_item_scores = col_series[rest_valid_mask]
        valid_totals = total_scores[rest_valid_mask]
        valid_rests = rest_score[rest_valid_mask]

        # 1. Madde-Toplam Korelasyonu (r_it)
        if item_sd > 1e-12 and total_sd > 1e-12 and len(valid_item_scores) > 2:
            r_it, _ = stats.pearsonr(valid_item_scores, valid_totals)
        else:
            r_it = np.nan

        # 2. Düzeltilmiş Madde-Toplam Korelasyonu (r_ir / Madde-Kalan)
        rest_sd = float(valid_rests.std(ddof=1)) if len(valid_rests) > 1 else 0.0
        if item_sd > 1e-12 and rest_sd > 1e-12 and len(valid_item_scores) > 2:
            r_ir, _ = stats.pearsonr(valid_item_scores, valid_rests)
        else:
            r_ir = np.nan

        # 3. Nokta-Çift Serili Korelasyon (r_pbis)
        # İkili verilerde r_it ile özdeştir; düzeltilmiş nokta-çift serili r_ir'dir.
        r_pbis = r_it

        # 4. Çift-Serili Korelasyon (r_bis)
        # Formül: r_bis = ((X_bar_1 - X_bar_0) / S_X) * (p * q / y)
        if is_dichotomous and item_sd > 1e-12 and total_sd > 1e-12:
            group1_scores = total_scores[col_series == 1].dropna()
            group0_scores = total_scores[col_series == 0].dropna()
            if len(group1_scores) > 0 and len(group0_scores) > 0 and 0.001 < p_val < 0.999:
                mean1 = float(group1_scores.mean())
                mean0 = float(group0_scores.mean())
                # Standart normal eğri ordinatı (y)
                z_p = stats.norm.ppf(p_val)
                y_p = stats.norm.pdf(z_p)
                if y_p > 1e-6:
                    r_bis = ((mean1 - mean0) / total_sd) * ((p_val * q_val) / y_p)
                else:
                    r_bis = np.nan
            else:
                r_bis = np.nan
        else:
            r_bis = np.nan

        # 5. Üst-Alt %27 Grup Ayırt Edicilik İndeksi (D)
        upper_vals = col_series[upper_group_mask].dropna()
        lower_vals = col_series[lower_group_mask].dropna()

        if len(upper_vals) > 0 and len(lower_vals) > 0:
            upper_mean = float(upper_vals.mean())
            lower_mean = float(lower_vals.mean())
            if is_dichotomous:
                upper_p = upper_mean
                lower_p = lower_mean
                d_val = upper_p - lower_p
            else:
                upper_p = (upper_mean - theoretical_min_item) / item_range
                lower_p = (lower_mean - theoretical_min_item) / item_range
                d_val = (upper_mean - lower_mean) / item_range
        else:
            upper_p = np.nan
            lower_p = np.nan
            d_val = np.nan

        # 6. Madde Silindiğinde Test İstatistikleri
        deleted_mean = total_mean - item_mean
        deleted_var = float(valid_rests.var(ddof=1)) if len(valid_rests) > 1 else 0.0

        # Madde silindiğinde Cronbach Alfa
        # Kalan k-1 madde matrisi
        remaining_items = [c for c in item_columns if c != col]
        if len(remaining_items) >= 2:
            rem_df = item_df[remaining_items].dropna()
            if len(rem_df) > 1:
                rem_k = len(remaining_items)
                rem_item_vars = rem_df.var(ddof=1).sum()
                rem_total_var = rem_df.sum(axis=1).var(ddof=1)
                if rem_total_var > 1e-12:
                    alpha_deleted = (rem_k / (rem_k - 1)) * (1.0 - (rem_item_vars / rem_total_var))
                else:
                    alpha_deleted = 0.0
            else:
                alpha_deleted = np.nan
        else:
            alpha_deleted = np.nan

        # 7. Pedagojik ve Psikometrik Tanısal Bayraklar
        flags = []
        if item_sd < 1e-12:
            flags.append("Sabit Madde (Varyans=0)")
        if p_val < 0.20:
            flags.append("Çok Zor (p < 0.20)")
        elif p_val > 0.80:
            flags.append("Çok Kolay (p > 0.80)")

        # Ayırt edicilik kontrolü (r_ir veya D üzerinden)
        disc_val = r_ir if not np.isnan(r_ir) else d_val
        if not np.isnan(disc_val):
            if disc_val < 0.0:
                flags.append("Negatif Ayırt Edicilik (Acil İnceleme!)")
            elif disc_val < 0.20:
                flags.append("Zayıf Ayırt Edicilik (Düzeltilmeli/Çıkarılmalı)")
            elif 0.20 <= disc_val < 0.30:
                flags.append("Sınırda Ayırt Edicilik (Geliştirilebilir)")

        if overall_alpha is not None and not np.isnan(alpha_deleted):
            if alpha_deleted > (overall_alpha + 0.01):
                flags.append("Silindiğinde Alfa Artıyor")

        flag_text = " | ".join(flags) if flags else "İdeal / Kabul Edilebilir"

        item_results.append({
            "Madde": col,
            "N": n_valid,
            "Kayıp %": round(missing_pct, 2),
            "Ortalama": round(item_mean, 4),
            "SS": round(item_sd, 4),
            "Varyans": round(item_var, 4),
            "Güçlük (p)": round(p_val, 4),
            "q (1-p)": round(q_val, 4),
            "r_it": round(r_it, 4) if not np.isnan(r_it) else np.nan,
            "r_ir (Düzeltilmiş)": round(r_ir, 4) if not np.isnan(r_ir) else np.nan,
            "r_pbis": round(r_pbis, 4) if not np.isnan(r_pbis) else np.nan,
            "r_bis": round(r_bis, 4) if not np.isnan(r_bis) else np.nan,
            "D (%27)": round(d_val, 4) if not np.isnan(d_val) else np.nan,
            "Üst Grup p": round(upper_p, 4) if not np.isnan(upper_p) else np.nan,
            "Alt Grup p": round(lower_p, 4) if not np.isnan(lower_p) else np.nan,
            "Silindiğinde Ort": round(deleted_mean, 4),
            "Silindiğinde Varyans": round(deleted_var, 4),
            "Silindiğinde Alfa": round(alpha_deleted, 4) if not np.isnan(alpha_deleted) else np.nan,
            "Tanısal Bayrak": flag_text
        })

    summary_df = pd.DataFrame(item_results)

    meta = {
        "n_examinees": n_examinees,
        "k_items": k_items,
        "n_upper_group": n_upper,
        "n_lower_group": n_lower,
        "p27_cutoff": p27_cutoff,
        "p73_cutoff": p73_cutoff,
        "mean_difficulty": round(float(summary_df["Güçlük (p)"].dropna().mean()), 4) if len(summary_df) > 0 else np.nan,
        "mean_discrimination_rir": round(float(summary_df["r_ir (Düzeltilmiş)"].dropna().mean()), 4) if len(summary_df) > 0 else np.nan,
        "mean_discrimination_D": round(float(summary_df["D (%27)"].dropna().mean()), 4) if len(summary_df) > 0 else np.nan,
        "problematic_items_count": int(summary_df["Tanısal Bayrak"].apply(lambda x: 1 if x != "İdeal / Kabul Edilebilir" else 0).sum())
    }

    return summary_df, meta
