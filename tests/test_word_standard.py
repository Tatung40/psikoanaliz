"""One reference-2PL layout contract for every real analysis/export family."""
import io
import re
import hashlib
import json
from pathlib import Path
import pandas as pd
from docx import Document
from docx.oxml.ns import qn
from reports.academic_docx import SECTION_ORDER,standardize_document,build_academic_docx


def assert_word_standard(doc):
    assert [p.text for p in doc.paragraphs if p.style.name=='Title']==['Bulgular ve Yorumlar — Sayısal Rapor']
    headings=[p.text for p in doc.paragraphs if p.style.name=='Heading 1']
    assert headings==[s for s in SECTION_ORDER if s in headings]
    assert len(headings)==len(set(headings))
    assert headings[0]==SECTION_ORDER[0] and headings[-1]==SECTION_ORDER[-1]
    assert SECTION_ORDER[5] in headings
    for label,count in [('Tablo',len(doc.tables)),('Şekil',len(doc.inline_shapes))]:
        numbers=[int(m.group(1)) for p in doc.paragraphs if (m:=re.fullmatch(label+r' (\d+)',p.text))]
        assert numbers==list(range(1,count+1)),(label,numbers,count)
    for section in doc.sections:
        assert section.page_width.inches==8.5 and section.page_height.inches==11
        assert section.top_margin.inches==section.bottom_margin.inches==1
        assert section.left_margin.inches==section.right_margin.inches==1.25
    for image in doc.inline_shapes:assert image.width.inches<=6
    assert doc.styles['Normal'].font.name=='Calibri' and doc.styles['Normal'].font.size.pt==11
    for name,size in [('Title',26),('Heading 1',14),('Heading 2',13),('Caption',9)]:
        assert doc.styles[name].font.size.pt==size
    for table in doc.tables:
        assert sum(column.width for column in table.columns)<=6*914400
        assert table.rows[0]._tr.xpath('./w:trPr/w:tblHeader')
        assert table._tbl.xpath('./w:tblPr/w:tblBorders/w:insideV')[0].get(qn('w:val'))=='nil'
    text='\n'.join(p.text for p in doc.paragraphs)
    assert 'Rapor standardı: 2pl-academic-v2' in text and 'Ortam paket sürümleri:' in text
    assert 'Bilimsel sınırlılıklar ve uyarılar' in text


def test_existing_2pl_document_keeps_curves_and_moves_them_before_reproducibility():
    from unittest.mock import patch
    from tests.test_irt_2pl_academic_docx import _report
    from reports.irt_2pl_academic_docx import export_2pl_docx
    with patch('reports.irt_2pl_academic_docx.build_report',return_value=_report()):
        doc=Document(io.BytesIO(export_2pl_docx(object(),['M1','M2'])))
    assert_word_standard(doc)
    text='\n'.join(p.text for p in doc.paragraphs)
    assert len(doc.inline_shapes)==3 and len(doc.tables)==3
    assert 'M1 ICC' in text and 'M2 ICC' in text
    assert text.index('Madde karakteristik eğrileri (ICC)')<text.index(SECTION_ORDER[5])<text.index(SECTION_ORDER[7])
    assert '120' in text


def test_captured_and_catalog_figures_share_one_sequence():
    from core.academic_result_schema import build_result
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots();ax.plot([0,1],[.2,.7]);png=io.BytesIO();fig.savefig(png,format='png');plt.close(fig)
    result=build_result(family='IRT',model='2PL',n=3,k=1,
        tables={'category_curves':[{'Theta':0.,'Olasılık':.2},{'Theta':1.,'Olasılık':.7}]},
        metadata={'analysis_type':'item_calibration'})
    doc=Document(io.BytesIO(build_academic_docx(result,figures=[('Gerçek ekran grafiği',png.getvalue())])))
    assert_word_standard(doc);assert len(doc.inline_shapes)==2
    instructions=' '.join(x.text or '' for x in doc.element.xpath('.//w:instrText'))
    assert instructions.count('SEQ Figure')==2 and instructions.count('SEQ Table')==1


def test_legacy_formatting_keeps_actual_cells_images_and_scientific_warnings():
    doc=Document();doc.add_heading('Özgün KTK başlığı',0)
    doc.add_heading('Madde inceleme tablosu',1)
    table=doc.add_table(rows=2,cols=2);table.cell(0,0).text='Madde';table.cell(0,1).text='p'
    table.cell(1,0).text='Türkçe_İÖŞ';table.cell(1,1).text='0.0000234'
    doc.add_heading('Karar ve bilimsel sınırlar',1);doc.add_paragraph('NOT CERTIFIED; p/q raporlanmaz.')
    before=[[c.text for c in r.cells] for r in table.rows]
    standardize_document(doc,method='Kaydedilen yöntem',provenance={'Seed':'SAĞLANMADI'})
    doc=Document(io.BytesIO(_save(doc)));assert_word_standard(doc)
    assert before==[[c.text for c in r.cells] for r in doc.tables[0].rows]
    text='\n'.join(p.text for p in doc.paragraphs)
    assert 'NOT CERTIFIED; p/q raporlanmaz.' in text and 'SAĞLANMADI' in text
    assert 'Model uyumu' not in [p.text for p in doc.paragraphs if p.style.name=='Heading 1']
    assert 'Şekiller bölümü atlandı' in text


def _save(doc):
    out=io.BytesIO();doc.save(out);return out.getvalue()


def test_reference_contract_and_all_routes_are_documented():
    from core.example_datasets import analysis_catalog
    from core.academic_screen_results import domain
    root=Path(__file__).resolve().parents[1]
    contract=json.loads((root/'validation/WORD_REFERENCE_CONTRACT.json').read_text())
    assert contract['reference_sha256'] and contract['target_sections']==list(SECTION_ORDER)
    matrix=(root/'validation/ACADEMIC_WORD_PYTHON_COVERAGE.md').read_text()
    assert all('| '+p+' |' in matrix for p in analysis_catalog() if domain(p) is not None)
    assert 'reports.polytomous_dif_report' in matrix and 'reports.word_export' in matrix


def test_aggregate_report_never_fills_missing_statistics_with_zero_or_fake_ci():
    from reports.word_export import create_word_report
    frame=pd.DataFrame({'Madde':['İ1'],'N':[5],'Güçlük (p)':[.4],'r_it':[.3],
                        'r_ir (Düzeltilmiş)':[.2],'D (%27)':[.1],'Silindiğinde Alfa':[.7],'Tanısal Bayrak':['İncele']})
    doc=Document(create_word_report({'n_examinees':5,'k_items':1},frame,{'cronbach_alpha':.71}))
    assert_word_standard(doc)
    text='\n'.join(p.text for p in doc.paragraphs)
    assert 'ortalaması Raporlanmadı' in text and 'ortalaması 0' not in text
    cells=' '.join(c.text for t in doc.tables for r in t.rows for c in r.cells)
    assert 'Feldt (1969) %95 Alfa GA' not in cells
    assert 'Feldt alfa güven aralığı: geçerli sınırlar' in text


def test_live_2pl_template_upgrade_rebuilds_only_word_and_keeps_real_fit():
    from tests.test_python_mirt_2pl_word_live_flow import _app
    at,items=_app('2PL','irt_006')
    fit=at.session_state['python_mirt_fit_2PL']['outcome']
    at.session_state['python_mirt_2pl_word'].update(standard='old-format',docx=b'old-format')
    at.run();assert not at.exception
    assert at.session_state['python_mirt_fit_2PL']['outcome'] is fit
    doc=Document(io.BytesIO(at.session_state['python_mirt_2pl_word']['docx']))
    assert_word_standard(doc)
    assert len(doc.inline_shapes)>=len(items)+1


def test_unrendered_sources_do_not_consume_numbers_and_formatting_is_idempotent():
    from core.academic_result_schema import build_result
    result=build_result(family='IRT',model='2PL',n=3,k=1,
        tables={'category_curves':[],'information':[{'θ':0.,'Bilgi':1.}]},
        metadata={'analysis_type':'item_calibration'})
    doc=Document(io.BytesIO(build_academic_docx(result)));assert_word_standard(doc)
    assert len(doc.tables)==len(doc.inline_shapes)==1
    before=[p.text for p in doc.paragraphs];standardize_document(doc)
    assert before==[p.text for p in doc.paragraphs]
    assert len(doc.settings.element.findall(qn('w:updateFields')))==1
    assert all(len(t.rows[0]._tr.xpath('./w:trPr/w:tblHeader'))==1 for t in doc.tables)
