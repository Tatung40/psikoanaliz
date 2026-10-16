"""
PsikoAnaliz — Güvenirlik ve Ölçme Hatası Modülü (core/reliability.py)
Cronbach alfa, KR-20 (Alfa ile kesin sayısal eşitlik garantili), KR-21, 
İki-Yarı güvenirlik, Spearman-Brown, Guttman L4, McDonald Omega ve SEM.
Geliştirici: Doç. Dr. Mehmet Taha Eser
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from scipy import stats


def compute_reliability_indices(
    df: pd.DataFrame,
    item_columns: List[str],
    is_dichotomous: bool = True,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Test için kapsamlı güvenirlik katsayılarını ve ölçme hatası metriklerini hesaplar.
    
    KR-20 ve Cronbach Alfa Eşdeğerlik Prensibi:
    İkili verilerde örneklem varyansı (ddof=1) standardı kullanıldığında:
    s_i^2 = (N / (N - 1)) * p_i * q_i
    Bu sayede kayıp verisiz ikili veri setlerinde KR-20 katsayısı Cronbach Alfa ile 
    virgülden sonra en az 12 basamak hassasiyetle birebir özdeştir:
    |Alpha - KR20| < 1e-12
    """
    item_df = df[item_columns].apply(pd.to_numeric, errors="coerce").dropna()
    n_examinees, k_items = item_df.shape

    if k_items < 2:
        return {
            "error": "Güvenirlik hesaplaması için en az 2 madde gereklidir.",
            "k_items": k_items,
            "n_examinees": n_examinees
        }

    if n_examinees < 3:
        return {
            "error": "Güvenirlik hesaplaması için en az 3 geçerli katılımcı gereklidir.",
            "k_items": k_items,
            "n_examinees": n_examinees
        }

    # Madde varyansları (Örneklem varyansı: ddof=1)
    item_variances = item_df.var(ddof=1)
    sum_item_vars = float(item_variances.sum())

    # Toplam test puanı ve varyansı (ddof=1)
    total_scores = item_df.sum(axis=1)
    total_var = float(total_scores.var(ddof=1))
    total_sd = float(total_scores.std(ddof=1))
    total_mean = float(total_scores.mean())

    results: Dict[str, Any] = {
        "k_items": k_items,
        "n_examinees": n_examinees,
        "is_dichotomous": is_dichotomous,
        "total_mean": round(total_mean, 4),
        "total_variance": round(total_var, 4),
        "total_std": round(total_sd, 4)
    }

    # 1. Cronbach Alfa
    if total_var > 1e-12:
        alpha = (k_items / (k_items - 1)) * (1.0 - (sum_item_vars / total_var))
    else:
        alpha = 0.0
    results["cronbach_alpha"] = round(alpha, 6)

    # 2. Maddeler Arası Korelasyon Matrisi ve Ortalama Korelasyon
    corr_matrix = item_df.corr(method="pearson")
    # Köşegen dışındaki korelasyonları al
    mask_off_diag = ~np.eye(k_items, dtype=bool)
    off_diag_corrs = corr_matrix.values[mask_off_diag]
    clean_off_diag = off_diag_corrs[~np.isnan(off_diag_corrs)]

    if len(clean_off_diag) > 0:
        mean_inter_item_r = float(np.mean(clean_off_diag))
        min_inter_item_r = float(np.min(clean_off_diag))
        max_inter_item_r = float(np.max(clean_off_diag))
    else:
        mean_inter_item_r = 0.0
        min_inter_item_r = 0.0
        max_inter_item_r = 0.0

    results["mean_inter_item_r"] = round(mean_inter_item_r, 4)
    results["min_inter_item_r"] = round(min_inter_item_r, 4)
    results["max_inter_item_r"] = round(max_inter_item_r, 4)
    results["correlation_matrix"] = corr_matrix

    # 3. Standartlaştırılmış Cronbach Alfa (Spearman-Brown dönüşümü)
    if mean_inter_item_r > -1.0 / (k_items - 1):
        alpha_std = (k_items * mean_inter_item_r) / (1.0 + (k_items - 1) * mean_inter_item_r)
    else:
        alpha_std = 0.0
    results["standardized_alpha"] = round(alpha_std, 6)

    # 4. Kuder-Richardson 20 (KR-20)
    # Sadece ikili (0/1) veriler için
    if is_dichotomous:
        # Madde güçlükleri
        p_vec = item_df.mean()
        q_vec = 1.0 - p_vec
        # Örneklem varyansı ile tutarlı KR-20 formülü:
        # Sum(s_i^2) = (N / (N - 1)) * Sum(p_i * q_i)
        sum_pq_corrected = (n_examinees / (n_examinees - 1)) * float(np.sum(p_vec * q_vec))

        if total_var > 1e-12:
            kr20 = (k_items / (k_items - 1)) * (1.0 - (sum_pq_corrected / total_var))
        else:
            kr20 = 0.0
        results["kr20"] = round(kr20, 6)

        # Doğrulama Farkı (|Alfa - KR-20|)
        results["alpha_kr20_difference"] = abs(alpha - kr20)

        # 5. Kuder-Richardson 21 (KR-21)
        # Formül: (k / (k - 1)) * (1 - (Mean * (k - Mean)) / (k * Var))
        if total_var > 1e-12 and k_items > 1:
            kr21 = (k_items / (k_items - 1)) * (1.0 - (total_mean * (k_items - total_mean)) / (k_items * total_var))
        else:
            kr21 = 0.0
        results["kr21"] = round(kr21, 6)
    else:
        results["kr20"] = None
        results["kr21"] = None
        results["alpha_kr20_difference"] = None

    # 6. İki-Yarı (Split-Half) Güvenirlik
    # a) Tek-Çift (Odd-Even) Bölmesi
    odd_cols = [item_columns[i] for i in range(0, k_items, 2)]
    even_cols = [item_columns[i] for i in range(1, k_items, 2)]

    if len(odd_cols) > 0 and len(even_cols) > 0:
        half1_scores = item_df[odd_cols].sum(axis=1)
        half2_scores = item_df[even_cols].sum(axis=1)

        r_half, _ = stats.pearsonr(half1_scores, half2_scores)
        # Spearman-Brown düzeltmesi
        if r_half > -1.0:
            sb_odd_even = (2.0 * r_half) / (1.0 + r_half)
        else:
            sb_odd_even = 0.0

        # Guttman Split-Half Lambda 4
        s1_sq = float(half1_scores.var(ddof=1))
        s2_sq = float(half2_scores.var(ddof=1))
        if total_var > 1e-12:
            guttman_l4 = 2.0 * (1.0 - ((s1_sq + s2_sq) / total_var))
        else:
            guttman_l4 = 0.0

        results["split_half_odd_even_r"] = round(r_half, 4)
        results["spearman_brown_odd_even"] = round(sb_odd_even, 4)
        results["guttman_lambda4"] = round(guttman_l4, 4)
    else:
        results["split_half_odd_even_r"] = np.nan
        results["spearman_brown_odd_even"] = np.nan
        results["guttman_lambda4"] = np.nan

    # b) İlk Yarı - İkinci Yarı (First-Second Half) Bölmesi
    half_split_idx = k_items // 2
    first_cols = item_columns[:half_split_idx]
    second_cols = item_columns[half_split_idx:]
    if len(first_cols) > 0 and len(second_cols) > 0:
        first_half = item_df[first_cols].sum(axis=1)
        second_half = item_df[second_cols].sum(axis=1)
        r_first_second, _ = stats.pearsonr(first_half, second_half)
        if r_first_second > -1.0:
            sb_first_second = (2.0 * r_first_second) / (1.0 + r_first_second)
        else:
            sb_first_second = 0.0
        results["split_half_first_second_r"] = round(r_first_second, 4)
        results["spearman_brown_first_second"] = round(sb_first_second, 4)
    else:
        results["split_half_first_second_r"] = np.nan
        results["spearman_brown_first_second"] = np.nan

    # 7. McDonald Omega (Omega Total - Tek Faktörlü Model Kestirimi)
    # Temel bileşenler / tek faktör faktör analizi yükleri üzerinden
    try:
        eigenvals, eigenvecs = np.linalg.eigh(corr_matrix.values)
        # En büyük özdeğer ve özvektör
        max_idx = np.argmax(eigenvals)
        max_eigenval = eigenvals[max_idx]
        if max_eigenval > 0:
            loadings = eigenvecs[:, max_idx] * np.sqrt(max_eigenval)
            # Yüklerin yönünü pozitif yap
            if np.mean(loadings) < 0:
                loadings = -loadings
            sum_loadings = np.sum(loadings)
            sum_sq_loadings = (sum_loadings) ** 2
            # Hata varyansları (uniquenesses)
            uniqueness = np.maximum(0.0, 1.0 - loadings ** 2)
            sum_uniqueness = np.sum(uniqueness)

            denom = sum_sq_loadings + sum_uniqueness
            if denom > 1e-12:
                omega_total = float(sum_sq_loadings / denom)
            else:
                omega_total = np.nan
        else:
            omega_total = np.nan
    except Exception:
        omega_total = np.nan
    results["mcdonald_omega"] = round(omega_total, 4) if not np.isnan(omega_total) else None

    # 8. Ölçmenin Standart Hatası (SEM)
    # Referans güvenirlik katsayısı: Alfa (veya ikili ise KR-20)
    primary_rel = alpha
    if primary_rel <= 1.0 and primary_rel >= 0.0:
        sem = float(total_sd * np.sqrt(1.0 - primary_rel))
    else:
        # Negatif güvenirlik durumunda SEM tanımsız / teorik olarak geçersizdir
        sem = total_sd

    results["sem"] = round(sem, 4)

    # Gerçek Puan Güven Aralıkları (Bireysel hata marjı)
    # %95 (z = 1.96) ve %99 (z = 2.576)
    results["ci_95_margin"] = round(1.95996 * sem, 4)
    results["ci_99_margin"] = round(2.57583 * sem, 4)

    # 9. Cronbach Alfa için Feldt (1969) Güven Aralığı
    # df1 = N - 1, df2 = (N - 1) * (k - 1)
    if alpha > 0 and n_examinees > 3 and k_items > 1:
        df1 = n_examinees - 1
        df2 = (n_examinees - 1) * (k_items - 1)
        f_lower_crit = stats.f.ppf((1.0 - confidence_level) / 2.0, df1, df2)
        f_upper_crit = stats.f.ppf(1.0 - (1.0 - confidence_level) / 2.0, df1, df2)

        ci_lower = 1.0 - ((1.0 - alpha) * stats.f.ppf(1.0 - (1.0 - confidence_level) / 2.0, df2, df1))
        ci_upper = 1.0 - ((1.0 - alpha) * stats.f.ppf((1.0 - confidence_level) / 2.0, df2, df1))

        results["alpha_ci_lower"] = round(max(0.0, float(ci_lower)), 4)
        results["alpha_ci_upper"] = round(min(1.0, float(ci_upper)), 4)
        results["alpha_ci_level"] = int(confidence_level * 100)
    else:
        results["alpha_ci_lower"] = np.nan
        results["alpha_ci_upper"] = np.nan
        results["alpha_ci_level"] = int(confidence_level * 100)

    # Pedagojik Güvenirlik Değerlendirmesi
    if primary_rel >= 0.90:
        rel_eval = "Mükemmel Güvenirlik (Bireysel yüksek riskli kararlar için uygun)"
    elif primary_rel >= 0.80:
        rel_eval = "Çok İyi Güvenirlik (Grup ve bireysel değerlendirmeler için yeterli)"
    elif primary_rel >= 0.70:
        rel_eval = "Kabul Edilebilir Güvenirlik (Araştırma ve sınıf içi ölçmeler için yeterli)"
    elif primary_rel >= 0.60:
        rel_eval = "Sınırda / Düşük Güvenirlik (Maddeler revize edilmeli veya madde sayısı artırılmalı)"
    else:
        rel_eval = "Yetersiz Güvenirlik (Test güvenilir ölçüm yapmıyor, madde analizi ile revize edilmeli)"

    results["reliability_evaluation"] = rel_eval
    return results
