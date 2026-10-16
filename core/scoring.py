"""
PsikoAnaliz — Puanlama ve Ölçekleme Motoru (core/scoring.py)
Çoktan seçmeli test puanlama, ters kodlama, çok kategorili puanlama ve toplam puan hesaplama.
Geliştirici: Doç. Dr. Mehmet Taha Eser
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd


def score_dichotomous(
    df: pd.DataFrame,
    item_columns: List[str],
    answer_key: Dict[str, Any],
    case_sensitive: bool = False,
    missing_as_zero: bool = False,
    id_column: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Çoktan seçmeli ham yanıtları cevap anahtarıyla karşılaştırarak 0/1 şeklinde ikili (dichotomous) puanlar.
    
    Parametreler:
    - df: Ham yanıtları içeren DataFrame
    - item_columns: Madde sütunlarının listesi
    - answer_key: {madde_adı: doğru_seçenek} sözlüğü
    - case_sensitive: Büyük/küçük harf duyarlılığı (Varsayılan: False)
    - missing_as_zero: Boş/kayıp yanıtların 0 olarak puanlanması (True) veya NaN bırakılması (False)
    - id_column: Varsa katılımcı kimlik sütunu
    
    Dönüş: (puanlanmis_df, meta_rapor)
    """
    scored_df = pd.DataFrame(index=df.index)
    if id_column and id_column in df.columns:
        scored_df[id_column] = df[id_column]

    diagnostics: Dict[str, Any] = {
        "items_scored": len(item_columns),
        "total_examinees": len(df),
        "missing_key_items": [],
        "invalid_entries": {}
    }

    for col in item_columns:
        if col not in answer_key:
            diagnostics["missing_key_items"].append(col)
            continue

        raw_key = answer_key[col]
        raw_series = df[col]

        # Normalizasyon
        if not case_sensitive:
            key_str = str(raw_key).strip().upper()
            series_str = raw_series.astype(str).str.strip().str.upper()
        else:
            key_str = str(raw_key).strip()
            series_str = raw_series.astype(str).str.strip()

        # Doğru yanıt eşleşmesi (1 veya 0)
        is_correct = (series_str == key_str).astype(float)

        # Orijinal kayıp verileri koruma veya 0 yapma
        is_na = raw_series.isna() | (raw_series.astype(str).str.strip() == "") | (raw_series.astype(str).str.lower() == "nan")
        if missing_as_zero:
            is_correct[is_na] = 0.0
        else:
            is_correct[is_na] = np.nan

        scored_df[col] = is_correct

    return scored_df, diagnostics


def reverse_score_items(
    df: pd.DataFrame,
    reverse_items: List[str],
    theoretical_min: float,
    theoretical_max: float
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ters yönlü maddeleri kuramsal (teorik) sınırlara dayalı olarak yeniden kodlar:
    X_yeni = Min_teorik + Max_teorik - X_eski
    
    UYARI: Kesinlikle ampirik min/max kullanılmaz; kuramsal ölçek sınırları zorunludur.
    Örnek: 1-5 Likert ölçeğinde: 1 + 5 - X = 6 - X (5 -> 1, 4 -> 2, 3 -> 3, 2 -> 4, 1 -> 5)
    """
    if theoretical_min >= theoretical_max:
        raise ValueError(
            f"Kuramsal minimum ({theoretical_min}) kuramsal maksimumdan ({theoretical_max}) küçük olmalıdır."
        )

    result_df = df.copy()
    report: Dict[str, Any] = {
        "reversed_items": reverse_items,
        "theoretical_min": theoretical_min,
        "theoretical_max": theoretical_max,
        "out_of_bounds_warnings": {}
    }

    scale_sum = theoretical_min + theoretical_max

    for item in reverse_items:
        if item not in result_df.columns:
            continue

        col_vals = pd.to_numeric(result_df[item], errors="coerce")
        # Sınır dışı değer kontrolü
        out_of_bounds = col_vals.dropna()[(col_vals.dropna() < theoretical_min) | (col_vals.dropna() > theoretical_max)]
        if len(out_of_bounds) > 0:
            report["out_of_bounds_warnings"][item] = {
                "count": len(out_of_bounds),
                "values": out_of_bounds.unique().tolist()
            }

        # Ters kodlama formülü
        result_df[item] = scale_sum - col_vals

    return result_df, report


def validate_polytomous_data(
    df: pd.DataFrame,
    item_columns: List[str],
    theoretical_min: float,
    theoretical_max: float
) -> Dict[str, Any]:
    """
    Çok kategorili (Likert vb.) veri setinin sayısal değer aralığını ve kuramsal sınırlarını denetler.
    """
    report: Dict[str, Any] = {
        "is_valid": True,
        "item_stats": {},
        "out_of_bounds_items": {}
    }

    for col in item_columns:
        if col not in df.columns:
            continue
        col_numeric = pd.to_numeric(df[col], errors="coerce")
        valid_vals = col_numeric.dropna()
        min_val = float(valid_vals.min()) if len(valid_vals) > 0 else np.nan
        max_val = float(valid_vals.max()) if len(valid_vals) > 0 else np.nan
        unique_vals = sorted(valid_vals.unique().tolist())

        # Sınır aşımı kontrolü
        invalid_mask = (valid_vals < theoretical_min) | (valid_vals > theoretical_max)
        if invalid_mask.any():
            report["is_valid"] = False
            report["out_of_bounds_items"][col] = {
                "invalid_count": int(invalid_mask.sum()),
                "invalid_samples": valid_vals[invalid_mask].unique().tolist()
            }

        report["item_stats"][col] = {
            "min": min_val,
            "max": max_val,
            "unique_values": unique_vals,
            "missing_count": int(col_numeric.isna().sum())
        }

    return report


def calculate_total_scores(
    df: pd.DataFrame,
    item_columns: List[str],
    missing_rule: str = "allow_partial",
    min_valid_items_ratio: float = 0.8
) -> pd.Series:
    """
    Katılımcıların toplam test puanlarını hesaplar.
    
    missing_rule:
    - 'strict': Tek bir kayıp veri dahi varsa toplam puan NaN olur.
    - 'allow_partial': En az min_valid_items_ratio oranında madde yanıtlandıysa geçerli maddelerin toplamını alır.
    - 'zero_impute': Kayıp verileri 0 sayarak toplar.
    """
    sub = df[item_columns].apply(pd.to_numeric, errors="coerce")
    k = len(item_columns)

    if missing_rule == "strict":
        return sub.sum(axis=1, skipna=False)

    elif missing_rule == "zero_impute":
        return sub.fillna(0.0).sum(axis=1)

    elif missing_rule == "allow_partial":
        valid_counts = sub.notna().sum(axis=1)
        min_required = int(np.ceil(k * min_valid_items_ratio))
        total_sum = sub.sum(axis=1, skipna=True)
        # Yeterli maddesi olmayan katılımcılara NaN ata
        total_sum[valid_counts < min_required] = np.nan
        return total_sum

    else:
        return sub.sum(axis=1, skipna=True)
