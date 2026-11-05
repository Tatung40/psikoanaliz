"""Regression tests for validation/OUTPUT_SCIENTIFIC_AUDIT.md findings 1-4 (display/report only)."""
import io
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

from reports.academic_docx import _value
from ui.p_display import apa_p_frame, format_p, is_p_column, truncation_note


def test_p_values_never_display_as_zero_and_numbers_are_preserved():
    frame = pd.DataFrame({'Madde': ['M1', 'M2', 'M3', 'M4'], 'p': [2.3e-5, 0., .0449999, np.nan],
                          'p_adj': [1e-12, .2, .05, .9], 'q (BH)': [.0004, .3, .04, 1.],
                          'Güçlük (p)': [.0001, .4, .5, .6], 'DIF (düzeltilmiş p < alpha)': [True, False, True, False]})
    before = frame.copy(deep=True)
    shown = apa_p_frame(frame).round(4)
    pd.testing.assert_frame_equal(frame, before)                       # results/exports untouched
    assert list(shown['p'][:3]) == ['< .001', '< .001', '0.0450'] and pd.isna(shown['p'][3])  # computed zero also "< .001"
    assert list(shown['p_adj'][:2]) == ['< .001', '0.2000'] and shown['q (BH)'][0] == '< .001'
    assert shown['Güçlük (p)'][0] == .0001 and shown['DIF (düzeltilmiş p < alpha)'].dtype == bool
    assert [_value(v) for v in shown['p'][:2]] == ['< .001', '< .001']  # Word shows the screen value
    assert not is_p_column('Parametre') and not is_p_column('Güçlük (p)') and is_p_column('Bootstrap p')
    assert format_p(.0012) == '0.0012' and format_p(float('inf')) is None


def test_ctt_word_sections_keep_headings_with_their_tables_in_order():
    from reports.word_export import create_word_report
    frame = pd.DataFrame({'Madde': ['İ1', 'İ2'], 'N': [5, 5], 'Güçlük (p)': [.4, .6], 'r_it': [.3, .4],
                          'r_ir (Düzeltilmiş)': [.2, .3], 'D (%27)': [.1, .2], 'Silindiğinde Alfa': [.7, .6],
                          'Tanısal Bayrak': ['İncele', '']})
    doc = Document(io.BytesIO(create_word_report({'n_examinees': 5, 'k_items': 2}, frame,
                                                 {'cronbach_alpha': .71}).getvalue()))
    sequence, last = [], None
    for element in doc.element.body:
        if element.tag == qn('w:tbl'):
            sequence.append(('TABLE', last))
        elif element.tag == qn('w:p'):
            p = Paragraph(element, doc)
            if p.style.name == 'Heading 2' and p.text[:2] in ('1.', '2.', '3.', '4.', '5.'):
                last = p.text[:2]; sequence.append(('H', last))
    assert [s for k, s in sequence if k == 'H'] == ['1.', '2.', '3.', '4.', '5.']
    assert [s for k, s in sequence if k == 'TABLE'] == ['2.', '3.', '4.']  # item table under section 4


def test_truncated_person_tables_are_disclosed():
    assert truncation_note(500, 1200) == 'İlk 500 / 1200 kişi gösteriliyor; tam tablo CSV/XLSX dışa aktarımındadır.'
    assert truncation_note(200, 900, export='CSV').endswith('tam tablo CSV dışa aktarımındadır.')
    assert truncation_note(150, 150) == ''
    root = Path(__file__).resolve().parents[1] / 'ui'
    assert 'truncation_note(min(len(pers), 500), len(pers))' in (root / 'irt_rsm_view.py').read_text()
    assert 'truncation_note(min(len(persons),200),len(persons),export="CSV")' in (root / 'irt_b14_view.py').read_text()


def test_loadings_screen_and_word_receive_the_same_numbers():
    source = (Path(__file__).resolve().parents[1] / 'ui/dimensionality_localfit_view.py').read_text()
    assert '.style.format' not in source and 't1.dataframe(res["loadings"].round(3))' in source


def test_observed_score_dif_page_shows_apa_p_on_screen():
    from tests.test_rsm_dif_pages import _app
    rng = np.random.default_rng(4); n = 600
    g = rng.integers(0, 2, n); th = rng.standard_normal(n)
    x = (rng.random((n, 8)) < 1 / (1 + np.exp(-(th[:, None] - np.linspace(-1, 1, 8))))).astype(int)
    x[:, 0] = np.where(g == 1, (rng.random(n) < .9).astype(int), x[:, 0])   # planted strong DIF
    df = pd.DataFrame(x, columns=[f'M{j}' for j in range(8)]); df['grup'] = np.where(g, 'B', 'A')
    at = _app(df, 'MTK: Gözlenen Puan DIF', [f'M{j}' for j in range(8)], group='grup')
    at.button(key='cdif_run').click().run()
    assert not at.exception
    raw = at.session_state['cdif_result']['frame']
    shown = at.dataframe[0].value
    assert raw['p'].min() < .001 and shown['p'].iloc[int(raw['p'].idxmin())] == '< .001'
    assert pd.api.types.is_float_dtype(raw['p']) and not (shown['p'].astype(str) == '0.0').any()
