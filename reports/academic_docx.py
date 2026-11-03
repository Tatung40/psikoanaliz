"""Offline, deterministic Word report from an already validated academic result.

No API, RAG, numerical re-estimation, or fabricated interpretation.
"""
from io import BytesIO

from docx import Document
from docx.shared import Inches
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.academic_result_schema import validate_result
from core.academic_visual_catalog import build_visual_catalog


def _value(value):
    if value is None:
        return "Raporlanmadı"
    if isinstance(value, bool):
        return "Evet" if value else "Hayır"
    if isinstance(value, float):
        return format(value, ".6g")
    return str(value)


def _figure(rows, spec):
    """Plot only real numeric observations; never interpolate missing statistics."""
    x, y = spec["x"], spec["y"]
    points = [(r[x], r[y]) for r in rows
              if isinstance(r.get(x), (int, float))
              and not isinstance(r.get(x), bool)
              and isinstance(r.get(y), (int, float))
              and not isinstance(r.get(y), bool)]
    if not points:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot([p[0] for p in points], [p[1] for p in points], linewidth=1.3)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    stream = BytesIO()
    fig.savefig(stream, format="png", dpi=160)
    plt.close(fig)
    stream.seek(0)
    return stream


def build_academic_docx(result, *, max_rows_per_table=100, validator=None, title=None,
                        figures=(), apa7=False):
    """Return DOCX bytes; report all omitted rows explicitly, no external requests."""
    source = (validator or validate_result)(result)
    if type(max_rows_per_table) is not int or max_rows_per_table < 1:
        raise ValueError("max_rows_per_table must be positive")
    catalog_source={**source,'family':source['family'] if source['family'] in ('CTT','IRT','GT') else 'CTT'}
    manifest = build_visual_catalog(catalog_source)
    doc = Document()
    if apa7:
        from docx.shared import Pt
        doc.styles['Normal'].font.name='Times New Roman'
        doc.styles['Normal'].font.size=Pt(12)
        for section in doc.sections:
            section.top_margin=section.bottom_margin=section.left_margin=section.right_margin=Inches(1)
    doc.add_heading(title or "Bulgular ve Yorumlar — Sayısal Rapor", 0)
    doc.add_paragraph(f"Alan: {source['family']} | Model: {source['model']}")
    doc.add_paragraph(f"N = {source['sample']['n']} | Madde sayısı: {_value(source['sample']['k'])}")
    doc.add_paragraph("Bu belge hesaplanan sonuçları sunar; kaynak temelli yorum ve istatistiksel çıkarım üretmez.")
    if source.get('metadata',{}).get('sample_basis'):
        doc.add_paragraph('Örneklem kapsamı: '+source['metadata']['sample_basis'])
        doc.add_heading('Yöntem ve kestirim ayarları',level=1)
        for setting in source['metadata'].get('settings',[]):doc.add_paragraph(str(setting))
    if source.get('metadata',{}).get('findings'):
        doc.add_heading('Bulgular',level=1)
        for finding in source['metadata']['findings']:doc.add_paragraph(str(finding))
    if source["fit_statistics"]:
        doc.add_heading("Model uyumu", level=1)
        for key, value in source["fit_statistics"].items():
            if isinstance(value, (list, dict)):
                doc.add_paragraph(f"{key}: yapılandırılmış veri; ilgili tablolara bakınız.")
            else:
                doc.add_paragraph(f"{key}: {_value(value)}")
    doc.add_heading("Tablolar", level=1)
    for spec in manifest["tables"]:
        rows = source["tables"][spec["source_table"]]
        doc.add_paragraph(spec["apa_label"], style="Normal")
        doc.add_paragraph(spec["title"])
        if not rows:
            doc.add_paragraph("Tabloda gözlem bulunmuyor.")
            continue
        columns = list(dict.fromkeys(k for row in rows for k in row))
        table = doc.add_table(rows=1, cols=len(columns))
        table.style = 'Table Grid'  # Visible horizontal and vertical borders in Word.
        if apa7:
            from docx.oxml import OxmlElement
            from docx.oxml.ns import qn
            borders=OxmlElement('w:tblBorders')
            for side in ('top','bottom','left','right','insideH','insideV'):
                border=OxmlElement('w:'+side);border.set(qn('w:val'),'single' if side in ('top','bottom') else 'nil')
                border.set(qn('w:sz'),'6');borders.append(border)
            table._tbl.tblPr.append(borders)
        for cell, name in zip(table.rows[0].cells, columns):
            cell.text = str(name)
        for row in rows[:max_rows_per_table]:
            cells = table.add_row().cells
            for cell, name in zip(cells, columns):
                cell.text = _value(row.get(name))
        if len(rows) > max_rows_per_table:
            suffix='ekrandaki tam tablo ve özgün analiz dışa aktarımlarına bakınız.' if apa7 else "özgün JSON'da korunur."
            doc.add_paragraph(f"Not. {len(rows)-max_rows_per_table} satır belge boyutu nedeniyle gösterilmedi; {suffix}")
    doc.add_heading("Şekiller", level=1)
    for number,(caption,png) in enumerate(figures,1):
        doc.add_paragraph(f'Şekil {number}');doc.add_paragraph(caption)
        doc.add_picture(BytesIO(png),width=Inches(5.8))
        doc.add_paragraph('Not. Ekranda gösterilen aynı gözlemlerden; ek istatistik kestirilmedi.')
    for spec in manifest["figures"]:
        plot = _figure(source["tables"][spec["source_table"]], spec)
        if plot is None:
            doc.add_paragraph(f"{spec['apa_label']}: sayısal grafik noktaları yok; çizilmedi.")
            continue
        doc.add_paragraph(spec["apa_label"])
        doc.add_paragraph(spec["title"])
        doc.add_picture(plot, width=Inches(5.8))
        doc.add_paragraph(f"Not. Kaynak: {spec['source_table']}; x={spec['x']}, y={spec['y']}.")
    warnings = list(source["warnings"]) + manifest["warnings"]
    if warnings:
        doc.add_heading("Bilimsel sınırlılıklar ve uyarılar", level=1)
        for warning in warnings:
            doc.add_paragraph(str(warning), style="List Bullet")
    doc.add_heading("Tekrarlanabilirlik bilgisi", level=1)
    for key, value in source["provenance"].items():
        doc.add_paragraph(f"{key}: {_value(value)}")
    stream = BytesIO()
    doc.save(stream)
    return stream.getvalue()
