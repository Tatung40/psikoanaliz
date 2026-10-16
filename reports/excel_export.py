"""
PsikoAnaliz — Excel Raporlama Motoru (reports/excel_export.py)
Çok sayfalı, APA uyumlu, stillendirilmiş ve formül enjeksiyonuna karşı korumalı Excel ihracı.
Geliştirici: Doç. Dr. Mehmet Taha Eser
"""

import io
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


def sanitize_excel_cell(val: Any) -> Any:
    """
    Formül enjeksiyonunu (CSV/Excel Formula Injection - CWE-1236) engellemek için
    kullanıcı kaynaklı metinlerin başına '=' konulmasını önler.
    """
    if isinstance(val, str):
        if val.startswith(("=", "+", "-", "@", "\t", "\r")):
            return "'" + val
    return val


def create_excel_report(
    test_summary: Dict[str, Any],
    item_summary_df: pd.DataFrame,
    reliability_data: Dict[str, Any],
    distractor_data: Optional[Dict[str, Dict[str, Any]]] = None,
    frequency_df: Optional[pd.DataFrame] = None,
    test_title: str = "PsikoAnaliz — Test Değerlendirme Raporu"
) -> io.BytesIO:
    """
    Tüm psikometrik analiz sonuçlarını profesyonel, kurumsal ve akademik stillerle
    biçimlendirilmiş çok sayfalı bir Excel (.xlsx) çalışma kitabına aktarır.
    """
    wb = openpyxl.Workbook()
    # Varsayılan ilk sayfayı kaldır
    wb.remove(wb.active)

    # Ortak Stiller
    header_fill = PatternFill(start_color="1E3A8A", end_color="1E3A8A", fill_type="solid")
    header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
    title_font = Font(name="Segoe UI", size=15, bold=True, color="1E3A8A")
    subtitle_font = Font(name="Segoe UI", size=10, italic=True, color="475569")
    bold_font = Font(name="Segoe UI", size=10, bold=True, color="0F172A")
    regular_font = Font(name="Segoe UI", size=10, color="1E293B")
    
    thin_border_side = Side(border_style="thin", color="CBD5E1")
    cell_border = Border(left=thin_border_side, right=thin_border_side, top=thin_border_side, bottom=thin_border_side)
    alt_row_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")

    # =========================================================================
    # SAYFA 1: TEST VE GÜVENİRLİK ÖZETİ
    # =========================================================================
    ws1 = wb.create_sheet(title="Test Özeti")
    ws1.views.sheetView[0].showGridLines = True

    ws1.cell(row=1, column=1, value=test_title).font = title_font
    ws1.cell(row=2, column=1, value="Hazırlayan: Doç. Dr. Mehmet Taha Eser | PsikoAnaliz KTK Platformu").font = subtitle_font

    # Bölüm 1: Test Betimsel İstatistikleri
    ws1.cell(row=4, column=1, value="TEST BETİMSEL İSTATİSTİKLERİ").font = Font(name="Segoe UI", size=11, bold=True, color="1E3A8A")
    ws1.cell(row=5, column=1, value="İstatistik").font = header_font
    ws1.cell(row=5, column=1).fill = header_fill
    ws1.cell(row=5, column=2, value="Değer").font = header_font
    ws1.cell(row=5, column=2).fill = header_fill

    desc_rows = [
        ("Katılımcı Sayısı (N)", test_summary.get("n_examinees", "-")),
        ("Madde Sayısı (k)", test_summary.get("k_items", "-")),
        ("Puan Ortalaması (Mean)", test_summary.get("mean", "-")),
        ("Medyan (Ortanca)", test_summary.get("median", "-")),
        ("Mod (Tepe Değer)", test_summary.get("mode", "-")),
        ("Standart Sapma (s)", test_summary.get("std_dev", "-")),
        ("Örneklem Varyansı (s²)", test_summary.get("variance", "-")),
        ("Ortalamanın Standart Hatası (SEM_mean)", test_summary.get("sem_mean", "-")),
        ("Minimum Puan", test_summary.get("min", "-")),
        ("Maksimum Puan", test_summary.get("max", "-")),
        ("Ranj (Aralık)", test_summary.get("range", "-")),
        ("Çarpıklık (Skewness)", test_summary.get("skewness", "-")),
        ("Basıklık (Kurtosis)", test_summary.get("kurtosis", "-")),
        ("Çeyrekler Arası Ranj (IQR)", test_summary.get("iqr", "-")),
        ("Taban Etkisi (Floor %)", f"%{test_summary.get('floor_pct', 0.0):.2f}"),
        ("Tavan Etkisi (Ceiling %)", f"%{test_summary.get('ceiling_pct', 0.0):.2f}")
    ]

    r = 6
    for label, val in desc_rows:
        c1 = ws1.cell(row=r, column=1, value=label)
        c2 = ws1.cell(row=r, column=2, value=val)
        c1.font = regular_font
        c2.font = bold_font
        c1.border = cell_border
        c2.border = cell_border
        if r % 2 == 1:
            c1.fill = alt_row_fill
            c2.fill = alt_row_fill
        r += 1

    # Bölüm 2: Güvenirlik ve Ölçme Hatası İstatistikleri
    ws1.cell(row=4, column=4, value="GÜVENİRLİK VE ÖLÇME HATASI").font = Font(name="Segoe UI", size=11, bold=True, color="1E3A8A")
    ws1.cell(row=5, column=4, value="Katsayı / Metrik").font = header_font
    ws1.cell(row=5, column=4).fill = header_fill
    ws1.cell(row=5, column=5, value="Değer").font = header_font
    ws1.cell(row=5, column=5).fill = header_fill

    rel_rows = [
        ("Cronbach Alfa (α)", reliability_data.get("cronbach_alpha", "-")),
        ("Standartlaştırılmış Alfa", reliability_data.get("standardized_alpha", "-")),
        ("KR-20 Katsayısı (İkili Test)", reliability_data.get("kr20", "-") if reliability_data.get("kr20") is not None else "Çok Kategorili Veri"),
        ("KR-21 Katsayısı (Eşit Güçlük)", reliability_data.get("kr21", "-") if reliability_data.get("kr21") is not None else "-"),
        ("İki-Yarı (Tek-Çift) r", reliability_data.get("split_half_odd_even_r", "-")),
        ("Spearman-Brown (Tek-Çift)", reliability_data.get("spearman_brown_odd_even", "-")),
        ("Guttman Split-Half (λ4)", reliability_data.get("guttman_lambda4", "-")),
        ("McDonald Omega (ω_t)", reliability_data.get("mcdonald_omega", "-")),
        ("Maddeler Arası Ort. Korelasyon", reliability_data.get("mean_inter_item_r", "-")),
        ("Ölçmenin Standart Hatası (SEM)", reliability_data.get("sem", "-")),
        ("Gerçek Puan %95 GA Marjı", f"± {reliability_data.get('ci_95_margin', '-')}"),
        ("Alfa %95 Güven Aralığı (Feldt)", f"[{reliability_data.get('alpha_ci_lower', '-')}, {reliability_data.get('alpha_ci_upper', '-')}]"),
        ("Genel Güvenirlik Yorumu", reliability_data.get("reliability_evaluation", "-"))
    ]

    r = 6
    for label, val in rel_rows:
        c1 = ws1.cell(row=r, column=4, value=label)
        c2 = ws1.cell(row=r, column=5, value=val)
        c1.font = regular_font
        c2.font = bold_font
        c1.border = cell_border
        c2.border = cell_border
        if r % 2 == 1:
            c1.fill = alt_row_fill
            c2.fill = alt_row_fill
        r += 1

    # Sütun Genişlikleri Otomatik Ayarlama
    for col in [1, 2, 4, 5]:
        col_letter = get_column_letter(col)
        ws1.column_dimensions[col_letter].width = 32

    # =========================================================================
    # SAYFA 2: MADDE ANALİZİ TABLOSU
    # =========================================================================
    ws2 = wb.create_sheet(title="Madde Analizi")
    ws2.views.sheetView[0].showGridLines = True

    ws2.cell(row=1, column=1, value="KLASİK TEST KURAMI MADDE ANALİZİ İSTATİSTİKLERİ").font = title_font

    cols_order = [
        "Madde", "N", "Kayıp %", "Ortalama", "SS", "Varyans", 
        "Güçlük (p)", "q (1-p)", "r_it", "r_ir (Düzeltilmiş)", 
        "r_pbis", "r_bis", "D (%27)", "Üst Grup p", "Alt Grup p", 
        "Silindiğinde Ort", "Silindiğinde Alfa", "Tanısal Bayrak"
    ]

    # Başlık Satırı
    for c_idx, col_name in enumerate(cols_order, 1):
        cell = ws2.cell(row=3, column=c_idx, value=col_name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")

    # Veri Satırları
    for r_idx, (_, row) in enumerate(item_summary_df.iterrows(), 4):
        for c_idx, col_name in enumerate(cols_order, 1):
            val = row.get(col_name, np.nan)
            if pd.isna(val):
                cell_val = "-"
            else:
                cell_val = sanitize_excel_cell(val)

            cell = ws2.cell(row=r_idx, column=c_idx, value=cell_val)
            cell.font = regular_font
            cell.border = cell_border

            # Hizalama
            if col_name in ["Madde", "Tanısal Bayrak"]:
                cell.alignment = Alignment(horizontal="left", vertical="center")
            else:
                cell.alignment = Alignment(horizontal="right", vertical="center")

            # Alternatif Satır Rengi
            if r_idx % 2 == 1:
                cell.fill = alt_row_fill

            # Tanısal Bayrak Renklendirme
            if col_name == "Tanısal Bayrak" and cell_val != "İdeal / Kabul Edilebilir" and cell_val != "-":
                cell.font = Font(name="Segoe UI", size=10, bold=True, color="B91C1C")
                cell.fill = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")

    for col in range(1, len(cols_order) + 1):
        col_letter = get_column_letter(col)
        ws2.column_dimensions[col_letter].width = 16
    ws2.column_dimensions["A"].width = 12
    ws2.column_dimensions[get_column_letter(len(cols_order))].width = 35

    # =========================================================================
    # SAYFA 3: ÇELDİRİCİ ANALİZİ (Varsa)
    # =========================================================================
    if distractor_data:
        ws3 = wb.create_sheet(title="Çeldirici Analizi")
        ws3.views.sheetView[0].showGridLines = True
        ws3.cell(row=1, column=1, value="ÇOKTAN SEÇMELİ ÇELDİRİCİ VE SEÇENEK ANALİZİ").font = title_font

        dist_headers = ["Madde", "Seçenek", "Doğru Cevap", "Frekans (n)", "Yüzde (%)", "Üst Grup %", "Alt Grup %", "Ayırt Edicilik (D_opt)", "r_pbis (Kalan)", "Değerlendirme"]
        for c_idx, h in enumerate(dist_headers, 1):
            cell = ws3.cell(row=3, column=c_idx, value=h)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")

        cur_row = 4
        for item_name, d_res in distractor_data.items():
            df_opts = d_res.get("options_table", pd.DataFrame())
            if df_opts.empty:
                continue
            for _, opt_row in df_opts.iterrows():
                ws3.cell(row=cur_row, column=1, value=item_name).font = bold_font
                ws3.cell(row=cur_row, column=2, value=opt_row["Seçenek"]).font = regular_font
                ws3.cell(row=cur_row, column=3, value=opt_row["Doğru Cevap"]).font = regular_font
                ws3.cell(row=cur_row, column=4, value=opt_row["Frekans (n)"]).font = regular_font
                ws3.cell(row=cur_row, column=5, value=f"%{opt_row['Yüzde (%)']:.1f}").font = regular_font
                ws3.cell(row=cur_row, column=6, value=f"%{opt_row['Üst Grup %']:.1f}").font = regular_font
                ws3.cell(row=cur_row, column=7, value=f"%{opt_row['Alt Grup %']:.1f}").font = regular_font
                ws3.cell(row=cur_row, column=8, value=opt_row["Ayırt Edicilik (D_opt)"]).font = regular_font
                ws3.cell(row=cur_row, column=9, value=opt_row["r_pbis (Kalan)"] if not pd.isna(opt_row["r_pbis (Kalan)"]) else "-").font = regular_font
                eval_cell = ws3.cell(row=cur_row, column=10, value=opt_row["Değerlendirme"])
                eval_cell.font = regular_font

                if "TERS ÇALIŞIYOR" in str(opt_row["Değerlendirme"]):
                    eval_cell.font = Font(name="Segoe UI", size=10, bold=True, color="B91C1C")
                    eval_cell.fill = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")

                for c in range(1, 11):
                    ws3.cell(row=cur_row, column=c).border = cell_border
                cur_row += 1

        for col in range(1, 11):
            ws3.column_dimensions[get_column_letter(col)].width = 16
        ws3.column_dimensions["J"].width = 38

    # =========================================================================
    # SAYFA 4: PUAN DAĞILIMI VE NORMLAR
    # =========================================================================
    if frequency_df is not None and not frequency_df.empty:
        ws4 = wb.create_sheet(title="Puan Dağılımı")
        ws4.views.sheetView[0].showGridLines = True
        ws4.cell(row=1, column=1, value="TEST PUANLARI FREKANS VE YÜZDELİK DAĞILIM TABLOSU").font = title_font

        freq_headers = list(frequency_df.columns)
        for c_idx, h in enumerate(freq_headers, 1):
            cell = ws4.cell(row=3, column=c_idx, value=h)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")

        for r_idx, (_, row) in enumerate(frequency_df.iterrows(), 4):
            for c_idx, h in enumerate(freq_headers, 1):
                cell = ws4.cell(row=r_idx, column=c_idx, value=row[h])
                cell.font = regular_font
                cell.border = cell_border
                cell.alignment = Alignment(horizontal="right", vertical="center")
                if r_idx % 2 == 1:
                    cell.fill = alt_row_fill

        for col in range(1, len(freq_headers) + 1):
            ws4.column_dimensions[get_column_letter(col)].width = 20

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output
