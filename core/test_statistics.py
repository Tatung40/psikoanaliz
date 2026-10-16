"""
PsikoAnaliz — Test Düzeyi İstatistikler ve Normlar Modülü (core/test_statistics.py)
Betimsel istatistikler, dağılım özellikleri, standart puanlar (z, T) ve norm tabloları.
Geliştirici: Doç. Dr. Mehmet Taha Eser
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from scipy import stats


def compute_test_summary(
    total_scores: pd.Series,
    k_items: int,
    theoretical_min_score: Optional[float] = None,
    theoretical_max_score: Optional[float] = None
) -> Dict[str, Any]:
    """
    Test toplam puanlarına ilişkin kapsamlı klasik test düzeyi betimsel istatistikleri hesaplar.
    
    Parametreler:
    - total_scores: Katılımcıların toplam puan serisi
    - k_items: Testteki toplam madde sayısı
    - theoretical_min_score: Kuramsal minimum alınabilecek test puanı (Opsiyonel)
    - theoretical_max_score: Kuramsal maksimum alınabilecek test puanı (Opsiyonel)
    
    Dönüş: Tüm betimsel istatistikleri ve norm göstergelerini içeren sözlük.
    """
    clean_scores = total_scores.dropna()
    n_examinees = len(clean_scores)
    n_missing = len(total_scores) - n_examinees

    if n_examinees == 0:
        return {
            "error": "Geçerli puan bulunamadı. Tüm katılımcı puanları kayıp veya tanımsız.",
            "n_examinees": 0,
            "k_items": k_items
        }

    # Temel eğilim ölçüleri
    mean_val = float(clean_scores.mean())
    median_val = float(clean_scores.median())
    mode_res = stats.mode(clean_scores, keepdims=True)
    mode_val = float(mode_res.mode[0]) if len(mode_res.mode) > 0 else np.nan

    min_val = float(clean_scores.min())
    max_val = float(clean_scores.max())
    score_range = max_val - min_val

    # Dağılım ve Değişkenlik ölçüleri (Örneklem varyansı: ddof=1)
    if n_examinees > 1:
        variance_val = float(clean_scores.var(ddof=1))
        sd_val = float(clean_scores.std(ddof=1))
        sem_mean = float(sd_val / np.sqrt(n_examinees))
    else:
        variance_val = 0.0
        sd_val = 0.0
        sem_mean = 0.0

    # Çarpıklık ve Basıklık (Önyargısız / Bias-corrected Fisher-Pearson)
    if n_examinees >= 3 and sd_val > 1e-12:
        skewness_val = float(stats.skew(clean_scores, bias=False))
    else:
        skewness_val = 0.0

    if n_examinees >= 4 and sd_val > 1e-12:
        # Excess kurtosis (Normal dağılım = 0)
        kurtosis_val = float(stats.kurtosis(clean_scores, bias=False))
    else:
        kurtosis_val = 0.0

    # Çeyreklikler ve Yüzdelikler
    q1 = float(np.percentile(clean_scores, 25))
    q2 = median_val
    q3 = float(np.percentile(clean_scores, 75))
    iqr = q3 - q1

    p10 = float(np.percentile(clean_scores, 10))
    p90 = float(np.percentile(clean_scores, 90))

    # Taban ve Tavan Etkisi
    # Kuramsal sınırlar verilmişse kuramsal sınırlara göre, yoksa gözlenen min/max'a göre
    t_min = theoretical_min_score if theoretical_min_score is not None else min_val
    t_max = theoretical_max_score if theoretical_max_score is not None else max_val

    floor_count = int((clean_scores <= t_min).sum())
    floor_pct = float(floor_count / n_examinees * 100) if n_examinees > 0 else 0.0

    ceiling_count = int((clean_scores >= t_max).sum())
    ceiling_pct = float(ceiling_count / n_examinees * 100) if n_examinees > 0 else 0.0

    # Ortalama güçlük (Puan ortalamasının madde sayısına oranı)
    mean_difficulty = (mean_val / (t_max - t_min)) if (t_max > t_min) else (mean_val / k_items if k_items > 0 else np.nan)

    return {
        "n_examinees": n_examinees,
        "n_missing": n_missing,
        "k_items": k_items,
        "mean": round(mean_val, 4),
        "median": round(median_val, 4),
        "mode": round(mode_val, 4),
        "min": round(min_val, 4),
        "max": round(max_val, 4),
        "range": round(score_range, 4),
        "variance": round(variance_val, 4),
        "std_dev": round(sd_val, 4),
        "sem_mean": round(sem_mean, 4),
        "skewness": round(skewness_val, 4),
        "kurtosis": round(kurtosis_val, 4),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "p10": round(p10, 4),
        "p90": round(p90, 4),
        "theoretical_min": t_min,
        "theoretical_max": t_max,
        "floor_count": floor_count,
        "floor_pct": round(floor_pct, 2),
        "ceiling_count": ceiling_count,
        "ceiling_pct": round(ceiling_pct, 2),
        "mean_difficulty": round(mean_difficulty, 4)
    }


def compute_standard_scores(
    scores: pd.Series,
    mean: Optional[float] = None,
    std_dev: Optional[float] = None
) -> pd.DataFrame:
    """
    Bireysel toplam puanları standart z ve T puanlarına ve yüzdelik sıralara dönüştürür.
    z = (X - Mean) / SD
    T = 50 + 10 * z
    """
    clean_scores = scores.astype(float)
    m = mean if mean is not None else float(clean_scores.mean())
    s = std_dev if std_dev is not None else float(clean_scores.std(ddof=1))

    df_std = pd.DataFrame({"raw_score": clean_scores})

    if s > 1e-12:
        df_std["z_score"] = (clean_scores - m) / s
        df_std["t_score"] = 50.0 + 10.0 * df_std["z_score"]
    else:
        df_std["z_score"] = 0.0
        df_std["t_score"] = 50.0

    # Yüzdelik sıra (Percentile rank)
    valid_mask = clean_scores.notna()
    percentiles = pd.Series(index=clean_scores.index, dtype=float)
    if valid_mask.sum() > 0:
        valid_vals = clean_scores[valid_mask]
        ranks = valid_vals.rank(pct=True, method="average") * 100.0
        percentiles[valid_mask] = ranks
    df_std["percentile_rank"] = percentiles

    return df_std


def compute_frequency_distribution(scores: pd.Series) -> pd.DataFrame:
    """
    Toplam puanların detaylı frekans, göreli frekans, kümülatif frekans ve yüzdelik sıra dağılımını üretir.
    """
    clean_scores = scores.dropna()
    if len(clean_scores) == 0:
        return pd.DataFrame()

    freq = clean_scores.value_counts().sort_index()
    total_n = len(clean_scores)

    dist_df = pd.DataFrame({
        "Puan": freq.index,
        "Frekans (f)": freq.values,
    })

    dist_df["Yüzde (%)"] = (dist_df["Frekans (f)"] / total_n * 100.0).round(2)
    dist_df["Kümülatif Frekans"] = dist_df["Frekans (f)"].cumsum()
    dist_df["Kümülatif Yüzde (%)"] = (dist_df["Kümülatif Frekans"] / total_n * 100.0).round(2)

    # Yüzdelik sıra: Puanın altındakiler + yarısı kuralı
    cum_prev = dist_df["Kümülatif Frekans"].shift(1).fillna(0)
    dist_df["Yüzdelik Sıra (PR)"] = (((cum_prev + (dist_df["Frekans (f)"] / 2.0)) / total_n) * 100.0).round(2)

    return dist_df
