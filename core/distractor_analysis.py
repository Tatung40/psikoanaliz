"""
PsikoAnaliz — Çeldirici Analizi Modülü (core/distractor_analysis.py)
Çoktan seçmeli testlerde seçenek dağılımı, üst-alt grup oranları, seçenek ayırt ediciliği ve çeldirici tanıları.
Geliştirici: Doç. Dr. Mehmet Taha Eser
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from scipy import stats


def analyze_distractors_for_item(
    raw_responses: pd.Series,
    correct_option: Any,
    total_scores: pd.Series,
    item_score: Optional[pd.Series] = None,
    non_functioning_threshold_pct: float = 5.0,
    options_list: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Tek bir çoktan seçmeli madde için detaylı seçenek ve çeldirici analizi yapar.
    
    Parametreler:
    - raw_responses: Katılımcıların ham seçenek yanıtları (A, B, C, D, E vb.)
    - correct_option: Doğru cevap seçeneği (Anahtar)
    - total_scores: Katılımcıların toplam test puanları
    - item_score: Katılımcının bu maddeden aldığı puan (Madde-kalan hesabı için)
    - non_functioning_threshold_pct: Çalışmayan çeldirici eşiği (Varsayılan: %5)
    - options_list: Varsa beklenen tüm seçeneklerin listesi (Örn: ['A', 'B', 'C', 'D', 'E'])
    
    Dönüş: Her seçenek için frekans, oran, üst-alt grup yüzdesi, D_opt, seçenek-kalan korelasyonu ve tanısal rapor.
    """
    key_str = str(correct_option).strip().upper()
    valid_mask = raw_responses.notna() & total_scores.notna()
    clean_resp = raw_responses[valid_mask].astype(str).str.strip().str.upper()
    clean_totals = total_scores[valid_mask]

    n_total = len(clean_resp)
    if n_total == 0:
        return {"error": "Geçerli yanıt bulunamadı.", "item_key": key_str}

    # Kalan puan (Rest score) hesabı
    if item_score is not None:
        clean_item_score = item_score[valid_mask].fillna(0.0)
        rest_scores = clean_totals - clean_item_score
    else:
        rest_scores = clean_totals

    # Üst ve Alt %27 Grubu belirleme
    p27_cutoff = float(np.percentile(clean_totals, 27))
    p73_cutoff = float(np.percentile(clean_totals, 73))

    upper_mask = clean_totals >= p73_cutoff
    lower_mask = clean_totals <= p27_cutoff

    n_upper = int(upper_mask.sum())
    n_lower = int(lower_mask.sum())

    # Benzersiz seçenekleri belirle
    observed_options = sorted(list(clean_resp.unique()))
    if options_list:
        all_options = sorted(list(set([str(o).strip().upper() for o in options_list] + observed_options)))
    else:
        all_options = observed_options

    if key_str not in all_options:
        all_options.append(key_str)
        all_options.sort()

    option_rows = []
    anomalies = []
    non_functioning_distractors = []
    inverted_distractors = []

    key_upper_pct = 0.0
    key_lower_pct = 0.0

    for opt in all_options:
        is_key = (opt == key_str)
        opt_mask = (clean_resp == opt)
        count_total = int(opt_mask.sum())
        pct_total = (count_total / n_total * 100.0) if n_total > 0 else 0.0

        count_upper = int((opt_mask & upper_mask).sum())
        pct_upper = (count_upper / n_upper * 100.0) if n_upper > 0 else 0.0

        count_lower = int((opt_mask & lower_mask).sum())
        pct_lower = (count_lower / n_lower * 100.0) if n_lower > 0 else 0.0

        # Seçenek Ayırt Ediciliği: D_opt = Pct_upper - Pct_lower
        d_opt = (pct_upper - pct_lower) / 100.0

        # Seçenek - Kalan Nokta-Çift Serili Korelasyon
        opt_binary = opt_mask.astype(float)
        if opt_binary.std() > 1e-12 and rest_scores.std() > 1e-12 and len(opt_binary) > 2:
            r_opt_rest, _ = stats.pearsonr(opt_binary, rest_scores)
        else:
            r_opt_rest = np.nan

        # Tanısal Bayraklandırma
        opt_status = []
        if is_key:
            key_upper_pct = pct_upper
            key_lower_pct = pct_lower
            if d_opt <= 0.0:
                opt_status.append("UYARI: Doğru cevap ayırt edici değil (D <= 0)")
                anomalies.append(f"Doğru cevap ({opt}) negatif veya sıfır ayırt ediciliğe sahip ({d_opt:.2f}).")
            else:
                opt_status.append("Doğru Cevap (Başarılı)")
        else:
            # Çeldirici değerlendirmesi
            if count_total == 0:
                opt_status.append("Hiç Seçilmedi (0)")
                non_functioning_distractors.append(opt)
            elif pct_total < non_functioning_threshold_pct:
                opt_status.append(f"Çalışmayan Çeldirici (< %{non_functioning_threshold_pct:.0f})")
                non_functioning_distractors.append(opt)
            elif d_opt > 0.05:
                # Üst grup alt gruptan belirgin şekilde daha çok seçmiş!
                opt_status.append("TERS ÇALIŞIYOR (Üst grup daha çok seçti!)")
                inverted_distractors.append(opt)
                anomalies.append(f"Çeldirici '{opt}' üst grup tarafından (%{pct_upper:.1f}) alt gruptan (%{pct_lower:.1f}) daha çok seçildi!")
            else:
                opt_status.append("İdeal Çeldirici")

        option_rows.append({
            "Seçenek": opt,
            "Doğru Cevap": "Evet" if is_key else "Hayır",
            "Frekans (n)": count_total,
            "Yüzde (%)": round(pct_total, 2),
            "Üst Grup n": count_upper,
            "Üst Grup %": round(pct_upper, 2),
            "Alt Grup n": count_lower,
            "Alt Grup %": round(pct_lower, 2),
            "Ayırt Edicilik (D_opt)": round(d_opt, 4),
            "r_pbis (Kalan)": round(r_opt_rest, 4) if not np.isnan(r_opt_rest) else np.nan,
            "Değerlendirme": " | ".join(opt_status)
        })

    # Genel Madde Düzeyinde Çeldirici Sağlığı
    overall_evaluation = "Çeldiriciler Başarılı"
    if anomalies:
        overall_evaluation = "KRİTİK SORUN: " + " ; ".join(anomalies)
    elif len(non_functioning_distractors) > 0:
        overall_evaluation = f"Zayıf Çeldirici(ler) Mevcut: {', '.join(non_functioning_distractors)}"

    df_table = pd.DataFrame(option_rows)

    return {
        "item_key": key_str,
        "n_total": n_total,
        "n_upper": n_upper,
        "n_lower": n_lower,
        "options_table": df_table,
        "non_functioning_distractors": non_functioning_distractors,
        "inverted_distractors": inverted_distractors,
        "anomalies": anomalies,
        "overall_evaluation": overall_evaluation
    }


def compute_all_items_distractors(
    raw_df: pd.DataFrame,
    item_columns: List[str],
    answer_key: Dict[str, Any],
    total_scores: pd.Series,
    scored_df: Optional[pd.DataFrame] = None,
    threshold_pct: float = 5.0
) -> Dict[str, Dict[str, Any]]:
    """
    Testteki tüm çoktan seçmeli maddeler için çeldirici analizlerini topluca gerçekleştirir.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for col in item_columns:
        if col not in raw_df.columns or col not in answer_key:
            continue
        key_val = answer_key[col]
        item_sc = scored_df[col] if (scored_df is not None and col in scored_df.columns) else None
        res = analyze_distractors_for_item(
            raw_responses=raw_df[col],
            correct_option=key_val,
            total_scores=total_scores,
            item_score=item_sc,
            non_functioning_threshold_pct=threshold_pct
        )
        results[col] = res
    return results
