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

REPORT_STANDARD_VERSION = "2pl-academic-v2"
SECTION_ORDER = ("Yöntem ve kestirim ayarları", "Bulgular", "Model uyumu", "Tablolar",
                 "Şekiller", "Bilimsel sınırlılıklar ve uyarılar", "Kaynaklar", "Tekrarlanabilirlik bilgisi")


def _section(text):
    """Map existing report headings, never infer a model/statistic from its data."""
    import re
    clean=re.sub(r'^\d+[.\s]+','',text).strip()
    if clean in SECTION_ORDER:return clean
    low=clean.casefold()
    if 'kaynak' in low:return 'Kaynaklar'
    if any(s in low for s in ('sınırl','sınır','uyarı','karar ve')):return SECTION_ORDER[5]
    if any(s in low for s in ('tekrarlan','yazılım','sürüm')):return SECTION_ORDER[7]
    if any(s in low for s in ('karakteristik','bilgi fonksiyon','grafi','şekil')):return 'Şekiller'
    if any(s in low for s in ('yöntem','yontem','giriş','test özellik')):return SECTION_ORDER[0]
    if 'model uyum' in low:return 'Model uyumu'
    if 'sonuç' in low or 'değerlendirme' in low:return 'Bulgular'
    return 'Tablolar'


def _number(paragraph, label, value):
    """Visible cached values plus Word SEQ fields; figures and tables have separate sequences."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    paragraph.clear();paragraph.style='Caption';paragraph.add_run(label+' ')
    for tag,text in (('begin',None),('instruction',' SEQ '+('Table' if label=='Tablo' else 'Figure')+' \\* ARABIC '),
                     ('separate',None),('value',str(value)),('end',None)):
        run=paragraph.add_run()
        if tag=='value':run.text=text;continue
        element=OxmlElement('w:instrText' if tag=='instruction' else 'w:fldChar')
        if tag=='instruction':element.set(qn('xml:space'),'preserve');element.text=text
        else:element.set(qn('w:fldCharType'),tag)
        run._r.append(element)
    paragraph.paragraph_format.keep_with_next=True


def standardize_document(doc, *, method=None, warnings=(), provenance=None):
    """Apply the reference 2PL hierarchy to an existing document without changing results.

    Reuses its XML blocks and image relationships. Legacy exporters keep their
    source content; only layout, section placement and captions are normalized.
    """
    import re
    import sys
    import importlib.metadata
    from docx.text.paragraph import Paragraph
    from docx.shared import Pt,RGBColor,Twips
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from core.reproducible import source_commit
    body=doc.element.body
    groups={name:[] for name in SECTION_ORDER};current=SECTION_ORDER[0]
    explicit=set(map(str,warnings));existing={p.text for p in doc.paragraphs}
    for element in list(body):
        if element.tag==qn('w:sectPr'):continue
        if element.tag==qn('w:p'):
            p=Paragraph(element,doc);text=p.text.strip()
            if p.style.name=='Title':
                if text!='Bulgular ve Yorumlar — Sayısal Rapor':
                    p.style='Subtitle';groups[SECTION_ORDER[0]].append(element)
                continue
            if p.style.name=='Heading 1':
                current=_section(text)
                if text in SECTION_ORDER:continue
                p.style='Heading 2'
            if text in explicit:groups[SECTION_ORDER[5]].append(element);continue
            if text.startswith('Şekil') or element.xpath('.//w:drawing'):current='Şekiller'
            elif re.match(r'^Tablo\s+\d+',text):current='Tablolar'
        elif element.tag==qn('w:tbl'):current='Tablolar'
        groups[current].append(element)
    def paragraph(text,style=None):return doc.add_paragraph(str(text),style)._p
    if method is not None and str(method) not in existing:groups[SECTION_ORDER[0]].append(paragraph(method))
    if not groups[SECTION_ORDER[0]]:
        groups[SECTION_ORDER[0]].append(paragraph('Yöntem ayrıntıları kaynak rapor girdisinde bulunmuyor; kestirim ayarı varsayılmadı.'))
    for note in warnings:
        if str(note) not in existing:groups[SECTION_ORDER[5]].append(paragraph(note))
    for section,available,reason in (
        ('Model uyumu',bool(groups['Model uyumu']),'kaynak çıktıda model-uyum istatistiği sağlanmadı'),
        ('Tablolar',bool(doc.tables),'kaynak çıktıda tablo sağlanmadı'),
        ('Şekiller',bool(doc.inline_shapes),'kaynak çıktıda gömülebilir grafik sağlanmadı')):
        if not available:
            groups[SECTION_ORDER[5]].extend(groups[section]);groups[section]=[]
            note=section+' bölümü atlandı: '+reason+'; yeni istatistik veya görsel üretilmedi.'
            if note not in existing:groups[SECTION_ORDER[5]].append(paragraph(note))
    caution='Rapor biçimlendirmesi bağımsız bilimsel doğrulama veya ek çıkarım yetkisi sağlamaz; kaynak analizdeki bilimsel kapılar korunur.'
    if caution not in existing:groups[SECTION_ORDER[5]].append(paragraph(caution))
    packages={}
    for name in ('python-docx','numpy','pandas','scipy','mirt','pingouin','generalizit'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    runtime={'Rapor standardı':REPORT_STANDARD_VERSION,'Python':sys.version.split()[0],
             'Ortam paket sürümleri':packages,'Kaynak commit':source_commit(),**(provenance or {})}
    for key,value in runtime.items():
        text=f'{key}: {_value(value)}'
        if text not in existing:groups[SECTION_ORDER[7]].append(paragraph(text))
    section_properties=body.find(qn('w:sectPr'))
    for element in list(body):body.remove(element)
    doc.add_heading('Bulgular ve Yorumlar — Sayısal Rapor',0)
    for name,elements in groups.items():
        if not elements:continue
        doc.add_heading(name,1)
        for element in elements:body.append(element)
    if section_properties is not None:body.append(section_properties)
    # Actual reference: US Letter; 1 inch top/bottom, 1.25 inch left/right;
    # title 26pt, H1 14pt, H2 13pt, normal 11pt, caption 9pt.
    for section in doc.sections:
        section.page_width=Inches(8.5);section.page_height=Inches(11)
        section.top_margin=section.bottom_margin=Inches(1)
        section.left_margin=section.right_margin=Inches(1.25)
    for name,size,font,color in (('Normal',11,'Calibri','000000'),('Title',26,'Cambria','17365D'),
            ('Heading 1',14,'Cambria','365F91'),('Heading 2',13,'Cambria','4F81BD'),('Caption',9,'Calibri','4F81BD')):
        style=doc.styles[name];style.font.name=font;style.font.size=Pt(size);style.font.color.rgb=RGBColor.from_string(color)
        style.paragraph_format.line_spacing=1.15
    counters={'Tablo':0,'Şekil':0}
    for element in list(body):
        kind='Tablo' if element.tag==qn('w:tbl') else 'Şekil' if element.xpath('.//w:drawing') else None
        if kind is None:continue
        counters[kind]+=1;previous=element.getprevious();caption=None;tail=''
        while previous is not None and previous.tag==qn('w:p'):
            p=Paragraph(previous,doc)
            if p.style.name.startswith('Heading'):break
            match=re.match(r'^'+kind+r'\s*(?:\d+\s*[:.]?|:)\s*(.*)$',p.text,re.S)
            if match:caption=p;tail=match.group(1).strip();break
            if previous.xpath('.//w:drawing'):break
            previous=previous.getprevious()
        if caption is None:
            caption=doc.add_paragraph();element.addprevious(caption._p)
        _number(caption,kind,counters[kind])
        if tail:
            title=doc.add_paragraph(tail);title.paragraph_format.keep_with_next=True
            for run in title.runs:run.italic=True
            caption._p.addnext(title._p)
    for table in doc.tables:
        table.style='Table Grid';table.autofit=True
        width=table._tbl.tblPr.find(qn('w:tblW'))
        if width is not None:width.set(qn('w:type'),'dxa');width.set(qn('w:w'),'8640')
        original=[column.width.twips if column.width else 1 for column in table.columns]
        remaining=8640
        for i,column in enumerate(table.columns):
            width=remaining if i==len(original)-1 else 8640*original[i]//sum(original)
            column.width=Twips(width);remaining-=width
        for row in table.rows:
            for i,cell in enumerate(row.cells):cell.width=table.columns[i].width
        borders=OxmlElement('w:tblBorders')
        for side in ('top','bottom','left','right','insideH','insideV'):
            border=OxmlElement('w:'+side);border.set(qn('w:val'),'single' if side in ('top','bottom') else 'nil');border.set(qn('w:sz'),'6');borders.append(border)
        for old in list(table._tbl.tblPr.findall(qn('w:tblBorders'))):table._tbl.tblPr.remove(old)
        table._tbl.tblPr.append(borders)
        for row_number,row in enumerate(table.rows):
            if row_number==0:
                properties=row._tr.get_or_add_trPr()
                for old in list(properties.findall(qn('w:tblHeader'))):properties.remove(old)
                header=OxmlElement('w:tblHeader');properties.append(header)
            for cell in row.cells:
                properties=cell._tc.get_or_add_tcPr()
                for old in list(properties.findall(qn('w:tcBorders'))):properties.remove(old)
                if row_number==0:
                    line=OxmlElement('w:tcBorders');bottom=OxmlElement('w:bottom');bottom.set(qn('w:val'),'single');bottom.set(qn('w:sz'),'6');line.append(bottom);properties.append(line)
                for p in cell.paragraphs:
                    for run in p.runs:run.font.name='Calibri';run.font.size=Pt(9);run.bold=row_number==0
    for picture in doc.inline_shapes:
        if picture.width>Inches(6):
            picture.height=int(picture.height*Inches(6)/picture.width);picture.width=Inches(6)
    for p in doc.paragraphs:
        size=doc.styles[p.style.name].font.size or Pt(11)
        if p.style.name.startswith('Heading'):p.paragraph_format.keep_with_next=True
        if p.text.startswith('Not.'):
            size=Pt(9)
            for run in p.runs:run.italic=True
        for run in p.runs:
            run.font.name=doc.styles[p.style.name].font.name or 'Calibri';run.font.size=size
            if p.style.name.startswith('Heading') or p.style.name=='Title':
                run.font.color.rgb=doc.styles[p.style.name].font.color.rgb
    for old in list(doc.settings.element.findall(qn('w:updateFields'))):doc.settings.element.remove(old)
    update=OxmlElement('w:updateFields');update.set(qn('w:val'),'true');doc.settings.element.append(update)
    return doc


def _value(value):
    if value is None:
        return "Raporlanmadı"
    if isinstance(value, bool):
        return "Evet" if value else "Hayır"
    if isinstance(value, float):
        import math
        if not math.isfinite(value):return 'Raporlanmadı'
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
                        figures=(), apa7=False, standardize=True):
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
        fit_keys={'loglik','loglikelihood','aic','bic','rmsea','srmr','cfi','tli','log_likelihood'}
        has_model_fit=any(str(key).casefold().replace(' ','') in fit_keys for key in source['fit_statistics'])
        doc.add_heading("Model uyumu" if has_model_fit else "Bulgular", level=1)
        for key, value in source["fit_statistics"].items():
            if isinstance(value, (list, dict)):
                doc.add_paragraph(f"{key}: yapılandırılmış veri; ilgili tablolara bakınız.")
            else:
                doc.add_paragraph(f"{key}: {_value(value)}")
    doc.add_heading("Tablolar", level=1)
    for spec in manifest["tables"]:
        rows = source["tables"][spec["source_table"]]
        if not rows:
            doc.add_paragraph(spec['title']+": kaynak tabloda gözlem yok; numaralı tablo oluşturulmadı.")
            continue
        doc.add_paragraph(spec["apa_label"], style="Normal")
        doc.add_paragraph(spec["title"])
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
            doc.add_paragraph(f"{spec['title']}: sayısal grafik noktaları yok; çizilmedi.")
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
    if standardize:
        standardize_document(doc,method=str(source.get('metadata',{}).get('model_metadata') or
                                            source.get('metadata',{}).get('settings') or
                                            {'Kaynak analiz':source['model']}))
    stream = BytesIO()
    doc.save(stream)
    return stream.getvalue()
