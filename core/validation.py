"""
PsikoAnaliz — Veri ve Model Doğrulama Modülü (core/validation.py)
Giriş verisi geçerliliği, sınır koşulları, sabit madde tespiti ve hata önleme kuralları.
Geliştirici: Doç. Dr. Mehmet Taha Eser
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd


def validate_analysis_inputs(
    df: pd.DataFrame,
    item_columns: List[str],
    is_dichotomous: bool = True,
    id_col: Optional[str] = None
) -> Tuple[bool, List[str], List[str]]:
    """
    Analiz öncesi veri matrisini kapsamlı şekilde doğrular.
    Dönüş: (is_valid, errors_list, warnings_list)
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df is None or len(df) == 0:
        errors.append("Veri seti boş. Lütfen geçerli bir veri seti yükleyin.")
        return False, errors, warnings

    if not item_columns or len(item_columns) == 0:
        errors.append("Hiçbir madde seçilmedi. En az bir madde seçilmelidir.")
        return False, errors, warnings

    if len(item_columns) < 2:
        warnings.append("Yalnızca 1 madde seçildi. Güvenirlik ve test analizleri için en az 2 madde önerilir.")

    missing_cols = [c for c in item_columns if c not in df.columns]
    if missing_cols:
        errors.append(f"Seçilen şu maddeler veri setinde bulunamadı: {missing_cols}")
        return False, errors, warnings

    # Katılımcı sayısı kontrolü
    n_examinees = len(df)
    if n_examinees < 5:
        warnings.append(f"Katılımcı sayısı çok düşük (N = {n_examinees}). Psikometrik kestirimler güvenilir olmayabilir.")

    # Sayısal kontrol ve varyans kontrolü
    item_df = df[item_columns].apply(pd.to_numeric, errors="coerce")

    # Sabit madde kontrolü (Varyans = 0)
    for col in item_columns:
        valid_vals = item_df[col].dropna()
        if len(valid_vals) == 0:
            warnings.append(f"'{col}' maddesinde hiç geçerli veri yok (tamamen kayıp).")
        elif valid_vals.nunique() <= 1:
            warnings.append(f"'{col}' maddesinde tüm katılımcılar aynı puanı almıştır (Sabit Madde, Varyans = 0).")

    # İkili veri kontrolü
    if is_dichotomous:
        for col in item_columns:
            vals = item_df[col].dropna().unique()
            non_binary = [v for v in vals if v not in [0, 1, 0.0, 1.0]]
            if len(non_binary) > 0:
                warnings.append(f"'{col}' ikili madde olarak tanımlandı ancak [0, 1] dışında değerler içeriyor: {non_binary[:5]}")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def sanitize_numeric(val: Any, default: float = np.nan) -> float:
    """
    Sayısal değerleri güvenli şekilde dönüştürür; sonsuzluk (inf) veya tanımsız (nan) durumlarını yakalar.
    """
    try:
        f_val = float(val)
        if np.isinf(f_val) or np.isnan(f_val):
            return default
        return f_val
    except (ValueError, TypeError):
        return default
