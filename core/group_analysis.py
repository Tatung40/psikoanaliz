"""
PsikoAnaliz — Grup Bazlı Analizler ve Karşılaştırma Modülü (core/group_analysis.py)
Grup düzeyinde test ve madde istatistikleri, t-testi, Welch t, Cohen's d, ANOVA ve KTK madde adillik taraması.
Geliştirici: Doç. Dr. Mehmet Taha Eser
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from scipy import stats


def compare_test_scores_by_group(
    df: pd.DataFrame,
    group_col: str,
    total_scores: pd.Series
) -> Dict[str, Any]:
    """
    Toplam test puanlarını kategorik grup değişkenine göre karşılaştırır.
    2 grup varsa: Bağımsız örneklem t-testi, Welch t-testi, Levene varyans homojenlik testi, Cohen's d.
    3+ grup varsa: Tek Yönlü ANOVA (F testi), Eta-kare (eta^2).
    """
    clean_mask = total_scores.notna() & df[group_col].notna()
    sub_df = pd.DataFrame({
        "group": df.loc[clean_mask, group_col].astype(str),
        "score": total_scores[clean_mask]
    })

    groups = sorted(sub_df["group"].unique().tolist())
    n_groups = len(groups)

    if n_groups < 2:
        return {
            "error": "Grup karşılaştırması için grupta en az 2 farklı kategori bulunmalıdır.",
            "groups_found": groups
        }

    # Grup betimsel tablosu
    group_summary = []
    group_data_list = []

    for g in groups:
        g_scores = sub_df[sub_df["group"] == g]["score"]
        n_g = len(g_scores)
        m_g = float(g_scores.mean()) if n_g > 0 else np.nan
        s_g = float(g_scores.std(ddof=1)) if n_g > 1 else 0.0
        group_data_list.append(g_scores)
        group_summary.append({
            "Grup": g,
            "N": n_g,
            "Ortalama": round(m_g, 4),
            "Standart Sapma": round(s_g, 4),
            "Medyan": round(float(g_scores.median()), 4) if n_g > 0 else np.nan,
            "Min": round(float(g_scores.min()), 4) if n_g > 0 else np.nan,
            "Max": round(float(g_scores.max()), 4) if n_g > 0 else np.nan
        })

    summary_df = pd.DataFrame(group_summary)
    results: Dict[str, Any] = {
        "group_variable": group_col,
        "n_groups": n_groups,
        "groups": groups,
        "group_summary_table": summary_df
    }

    if n_groups == 2:
        g1_scores = group_data_list[0]
        g2_scores = group_data_list[1]
        n1, n2 = len(g1_scores), len(g2_scores)
        m1, m2 = float(g1_scores.mean()), float(g2_scores.mean())
        s1, s2 = float(g1_scores.std(ddof=1)), float(g2_scores.std(ddof=1))

        # Levene Testi (Varyans eşitliği)
        levene_stat, levene_p = stats.levene(g1_scores, g2_scores)

        # Standart Student t-testi
        t_stat_std, p_val_std = stats.ttest_ind(g1_scores, g2_scores, equal_var=True)

        # Welch t-testi (Varyanslar eşit varsayılmadığında)
        t_stat_welch, p_val_welch = stats.ttest_ind(g1_scores, g2_scores, equal_var=False)

        # Cohen's d (Birleşik standart sapma ile)
        if n1 + n2 > 2:
            s_pooled = np.sqrt(((n1 - 1) * (s1 ** 2) + (n2 - 1) * (s2 ** 2)) / (n1 + n2 - 2))
            cohens_d = (m1 - m2) / s_pooled if s_pooled > 1e-12 else 0.0
        else:
            cohens_d = 0.0

        # Cohen's d etki büyüklüğü yorumu
        abs_d = abs(cohens_d)
        if abs_d < 0.20:
            d_interpretation = "İhmal edilebilir / Çok küçük etki"
        elif abs_d < 0.50:
            d_interpretation = "Küçük etki (Small effect)"
        elif abs_d < 0.80:
            d_interpretation = "Orta etki (Medium effect)"
        else:
            d_interpretation = "Geniş / Büyük etki (Large effect)"

        results["comparison_type"] = "2_groups_ttest"
        results["levene_stat"] = round(float(levene_stat), 4)
        results["levene_p"] = round(float(levene_p), 4)
        results["equal_variance_assumed"] = bool(levene_p > 0.05)
        results["student_t"] = round(float(t_stat_std), 4)
        results["student_p"] = round(float(p_val_std), 4)
        results["welch_t"] = round(float(t_stat_welch), 4)
        results["welch_p"] = round(float(p_val_welch), 4)
        results["cohens_d"] = round(float(cohens_d), 4)
        results["effect_size_interpretation"] = d_interpretation

    else:
        # Tek Yönlü ANOVA
        f_stat, anova_p = stats.f_oneway(*group_data_list)
        # Eta-kare hesabı: SS_between / SS_total
        grand_mean = float(sub_df["score"].mean())
        ss_total = float(((sub_df["score"] - grand_mean) ** 2).sum())
        ss_between = sum([len(g_data) * ((float(g_data.mean()) - grand_mean) ** 2) for g_data in group_data_list])
        eta_squared = (ss_between / ss_total) if ss_total > 1e-12 else 0.0

        results["comparison_type"] = "multi_group_anova"
        results["f_statistic"] = round(float(f_stat), 4)
        results["anova_p"] = round(float(anova_p), 4)
        results["eta_squared"] = round(float(eta_squared), 4)

    return results


def compute_group_item_fairness(
    df: pd.DataFrame,
    group_col: str,
    item_columns: List[str],
    reference_group: Optional[str] = None,
    focal_group: Optional[str] = None,
    delta_p_threshold: float = 0.15
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Klasik Test Kuramı (KTK) düzeyinde maddeler için betimsel grup güçlük farklarını (Delta p) hesaplar.
    
    ÖNEMLİ METODOLOJİK UYARI:
    Bu analiz ön tarama amaçlı bir KTK betimsel güçlük farkı incelemesidir (Delta p = p_ref - p_focal).
    Madde Yanlılığı (DIF / Değişen Madde Fonksiyonlaşması) için formel kanıt DEĞİLDİR.
    DIF için yetenek düzeyine göre koşullu (Mantel-Haenszel veya IRT tabanlı) analizler gereklidir.
    """
    valid_groups = df[group_col].dropna().unique().tolist()
    if len(valid_groups) < 2:
        return pd.DataFrame(), {"error": "En az 2 grup gereklidir."}

    ref = reference_group if reference_group in valid_groups else str(valid_groups[0])
    foc = focal_group if focal_group in valid_groups and focal_group != ref else str(valid_groups[1])

    ref_mask = (df[group_col].astype(str) == ref)
    foc_mask = (df[group_col].astype(str) == foc)

    ref_df = df[ref_mask][item_columns].apply(pd.to_numeric, errors="coerce")
    foc_df = df[foc_mask][item_columns].apply(pd.to_numeric, errors="coerce")

    n_ref = len(ref_df)
    n_foc = len(foc_df)

    item_rows = []
    flagged_count = 0

    for col in item_columns:
        p_ref = float(ref_df[col].dropna().mean()) if n_ref > 0 else np.nan
        p_foc = float(foc_df[col].dropna().mean()) if n_foc > 0 else np.nan

        if not np.isnan(p_ref) and not np.isnan(p_foc):
            delta_p = p_ref - p_foc
            abs_delta = abs(delta_p)

            # İki oran arasındaki fark için z testi
            p_pooled = (ref_df[col].dropna().sum() + foc_df[col].dropna().sum()) / (n_ref + n_foc)
            se_diff = np.sqrt(p_pooled * (1.0 - p_pooled) * (1.0 / n_ref + 1.0 / n_foc)) if 0 < p_pooled < 1 else 0.0

            if se_diff > 1e-12:
                z_stat = delta_p / se_diff
                p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
            else:
                z_stat = 0.0
                p_val = 1.0

            if abs_delta >= delta_p_threshold:
                flag = f"Önemli Güçlük Farkı (|Δp| >= {delta_p_threshold:.2f})"
                flagged_count += 1
            elif p_val < 0.05:
                flag = "İstatistiki Farklılık (p < .05, düşük etki)"
            else:
                flag = "Dengeli (Fark yok)"
        else:
            delta_p = np.nan
            z_stat = np.nan
            p_val = np.nan
            flag = "Yetersiz Veri"

        item_rows.append({
            "Madde": col,
            f"Referans ({ref}) p": round(p_ref, 4) if not np.isnan(p_ref) else np.nan,
            f"Odak ({foc}) p": round(p_foc, 4) if not np.isnan(p_foc) else np.nan,
            "Fark (Δp = p_ref - p_foc)": round(delta_p, 4) if not np.isnan(delta_p) else np.nan,
            "z Değeri": round(float(z_stat), 4) if not np.isnan(z_stat) else np.nan,
            "p Değeri": round(float(p_val), 4) if not np.isnan(p_val) else np.nan,
            "Ön Tarama Durumu": flag
        })

    fairness_df = pd.DataFrame(item_rows)
    meta = {
        "reference_group": ref,
        "focal_group": foc,
        "n_reference": n_ref,
        "n_focal": n_foc,
        "threshold_used": delta_p_threshold,
        "flagged_items_count": flagged_count,
        "methodological_note": (
            "Bu tablo KTK çerçevesinde tanımlayıcı güçlük farklarını (Δp) gösterir. "
            "Yetenek düzeyini kontrol eden formel Değişen Madde Fonksiyonlaşması (DIF) analizinin yerine geçmez."
        )
    }

    return fairness_df, meta
