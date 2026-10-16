"""
PsikoAnaliz — Veri İşleme Modülü (core/data_processing.py)
Dosya içe aktarma, kodlama ve ayırıcı tespiti, kayıp veri yönetimi ve veri doğrulama.
Geliştirici: Doç. Dr. Mehmet Taha Eser
"""

import io
import csv
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd


def detect_csv_properties(file_bytes: bytes) -> Tuple[str, str]:
    """
    CSV dosyasının karakter kodlamasını (encoding) ve ayırıcısını (delimiter) otomatik tespit eder.
    Özellikle Türkçe karakter içeren Windows-1254, ISO-8859-9 ve UTF-8 dosyalarını destekler.
    """
    encodings = ["utf-8-sig", "utf-8", "windows-1254", "iso-8859-9", "latin1", "cp1252"]
    detected_encoding = "utf-8"
    sample_text = ""

    for enc in encodings:
        try:
            sample_text = file_bytes[:8192].decode(enc)
            detected_encoding = enc
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    # Ayırıcı tespiti
    delimiter = ","
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample_text)
        delimiter = dialect.delimiter
    except Exception:
        # Sniffer başarısız olursa frekans analizi yap
        semicolon_count = sample_text.count(";")
        comma_count = sample_text.count(",")
        tab_count = sample_text.count("\t")
        if semicolon_count > comma_count and semicolon_count > tab_count:
            delimiter = ";"
        elif tab_count > comma_count and tab_count > semicolon_count:
            delimiter = "\t"
        else:
            delimiter = ","

    return detected_encoding, delimiter


def load_dataset(
    file_source: Union[str, io.BytesIO, bytes],
    file_type: str = "csv",
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    CSV, Excel veya SPSS (.sav) formatındaki veri setini yükler.
    Dönüş: (pd.DataFrame, meta_bilgiler)
    """
    meta: Dict[str, Any] = {
        "format": file_type.lower(),
        "encoding": encoding,
        "delimiter": delimiter,
        "warnings": []
    }

    if isinstance(file_source, bytes):
        file_bytes = file_source
        buffer = io.BytesIO(file_bytes)
    elif isinstance(file_source, io.BytesIO):
        buffer = file_source
        file_bytes = file_source.getvalue()
    elif isinstance(file_source, str):
        with open(file_source, "rb") as f:
            file_bytes = f.read()
        buffer = io.BytesIO(file_bytes)
    else:
        # Streamlit UploadedFile objesi
        file_bytes = file_source.read()
        buffer = io.BytesIO(file_bytes)

    fmt = file_type.lower().replace(".", "")

    if fmt == "csv":
        if not encoding or not delimiter:
            auto_enc, auto_delim = detect_csv_properties(file_bytes)
            encoding = encoding or auto_enc
            delimiter = delimiter or auto_delim
        meta["encoding"] = encoding
        meta["delimiter"] = delimiter
        df = pd.read_csv(io.StringIO(file_bytes.decode(encoding, errors="replace")), sep=delimiter)

    elif fmt in ["xlsx", "xls"]:
        df = pd.read_excel(buffer)
        meta["sheets"] = "Default"

    elif fmt == "sav":
        try:
            import pyreadstat
            df, sav_meta = pyreadstat.read_sav(buffer)
            meta["column_labels"] = sav_meta.column_names_to_labels
            meta["variable_value_labels"] = sav_meta.variable_value_labels
        except ImportError:
            raise ImportError(
                "SPSS (.sav) dosyalarını okumak için 'pyreadstat' kütüphanesi gereklidir. "
                "Lütfen 'pip install pyreadstat' komutunu çalıştırınız veya verinizi Excel/CSV olarak yükleyiniz."
            )
    else:
        raise ValueError(f"Desteklenmeyen dosya formatı: {fmt}. Lütfen CSV, XLSX veya SAV yükleyin.")

    # Sütun adlarındaki boşlukları temizle
    df.columns = [str(c).strip() for c in df.columns]
    return df, meta


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Yüklenen veri setinin temel özelliklerini (satır, sütun, kayıp veri, tipler) raporlar.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    missing_cells = int(df.isna().sum().sum())
    total_cells = total_rows * total_cols
    missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0.0

    col_info = []
    for col in df.columns:
        col_series = df[col]
        missing_count = int(col_series.isna().sum())
        missing_rate = (missing_count / total_rows * 100) if total_rows > 0 else 0.0
        unique_vals = col_series.dropna().unique()
        unique_count = len(unique_vals)

        # Sayısal mı veya kategorik mi
        is_numeric = pd.api.types.is_numeric_dtype(col_series)
        col_info.append({
            "column": col,
            "is_numeric": is_numeric,
            "missing_count": missing_count,
            "missing_pct": round(missing_rate, 2),
            "unique_count": unique_count,
            "sample_values": list(unique_vals[:5])
        })

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "missing_cells": missing_cells,
        "missing_pct": round(missing_pct, 2),
        "columns": col_info
    }


def handle_missing_data(
    df: pd.DataFrame,
    method: str = "keep",
    item_columns: Optional[List[str]] = None,
    zero_value: float = 0.0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Kayıp veri yönetim protokolünü uygular.
    Metotlar:
    - 'keep': Kayıp verileri (NaN) olduğu gibi bırakır.
    - 'listwise': Madde sütunlarında herhangi bir kayıp verisi olan katılımcıları tamamen siler.
    - 'zero': Yanıtlanmayan/boş bırakılan maddelere sıfır (0) puan atar (Başarı testleri için standart).
    - 'mean': Madde ortalaması ile doldurur (Yalnızca sayısal ve çok kategorili ölçekler için önerilir).
    """
    cols = item_columns if item_columns else list(df.columns)
    initial_rows = len(df)
    info: Dict[str, Any] = {
        "method": method,
        "initial_rows": initial_rows,
        "dropped_rows": 0,
        "final_rows": initial_rows,
        "imputed_cells": 0
    }

    result_df = df.copy()

    if method == "listwise":
        cleaned_df = result_df.dropna(subset=cols)
        info["dropped_rows"] = initial_rows - len(cleaned_df)
        info["final_rows"] = len(cleaned_df)
        return cleaned_df, info

    elif method == "zero":
        sub = result_df[cols]
        missing_count = int(sub.isna().sum().sum())
        result_df[cols] = sub.fillna(zero_value)
        info["imputed_cells"] = missing_count
        return result_df, info

    elif method == "mean":
        sub = result_df[cols]
        missing_count = int(sub.isna().sum().sum())
        for c in cols:
            if pd.api.types.is_numeric_dtype(result_df[c]):
                mean_val = result_df[c].mean()
                result_df[c] = result_df[c].fillna(mean_val)
        info["imputed_cells"] = missing_count
        return result_df, info

    elif method == "keep":
        return result_df, info

    else:
        raise ValueError(f"Geçersiz kayıp veri metodu: {method}. Seçenekler: 'keep', 'listwise', 'zero', 'mean'")


def check_duplicate_participants(df: pd.DataFrame, id_col: str) -> List[Any]:
    """
    Belirtilen ID sütununda yinelenen (mükerrer) katılımcı kimliklerini tespit eder.
    """
    if id_col not in df.columns:
        return []
    duplicates = df[df.duplicated(subset=[id_col], keep=False)][id_col].unique().tolist()
    return duplicates
