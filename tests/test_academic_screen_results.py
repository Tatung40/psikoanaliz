"""No-fit tests of reporting-only adapters: no invented estimates or inference."""
import io
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from docx import Document
from core.academic_screen_results import adapt_screen,validate_screen_result,convergence,domain
from core.reproducible import begin,finish
from core import descriptives
from reports.academic_docx import build_academic_docx


def test_new_family_keeps_domain_actual_cells_and_suppressed_statistics():
    x=pd.DataFrame({'M1':[0.,1.,1.],'M2':[1.,0.,1.]})
    recorder=begin('MTK: DIF Analiz Merkezi');descriptives.describe_items(x);finish()
    table=pd.DataFrame({'Item':['M1'],'χ²':[1.234567],'p':[np.nan],'q':[np.nan]})
    result=adapt_screen('MTK: DIF Analiz Merkezi',recorder,[('DIF',table)],{'N':3},['NOT CERTIFIED'],x)
    assert result['family']=='DIF' and result['tables']['DIF'][0]=={'Item':'M1','χ²':1.234567,'p':None,'q':None}
    doc=Document(io.BytesIO(build_academic_docx(result,validator=validate_screen_result,apa7=True)))
    cells=' '.join(c.text for t in doc.tables for row in t.rows for c in row.cells)
    assert '1.23457' in cells and 'Raporlanmadı' in cells
    assert any('NOT CERTIFIED' in p.text for p in doc.paragraphs)
    assert doc.styles['Normal'].font.name=='Times New Roman'
    assert 'insideV' in doc.tables[0]._tbl.xml and 'nil' in doc.tables[0]._tbl.xml
    assert 'Bulgular' in '\n'.join(p.text for p in doc.paragraphs)


def test_nested_failed_fit_is_not_a_valid_academic_result():
    value={'fit':{'converged':False}}
    recorder=SimpleNamespace(keep=[value],references={id(value):[0]})
    assert convergence(recorder) is False


def test_static_surface_uses_actual_grid_without_fitting():
    import plotly.graph_objects as go
    from ui.academic_word import _plotly_png
    png=_plotly_png('Gerçek yüzey',go.Figure(go.Surface(x=[0.,1.],y=[0.,1.],z=[[.2,.4],[.5,.8]])))[1]
    assert png.startswith(b'\x89PNG')
    # A figure is never partially reproduced when a trace has unsupported semantics.
    assert _plotly_png('Histogram',go.Figure([go.Histogram(x=[1,2]),go.Scatter(x=[1,2],y=[1,2])])) is None


def test_coverage_inventory_has_exactly_the_live_analysis_routes():
    from core.example_datasets import analysis_catalog
    from pathlib import Path
    text=(Path(__file__).resolve().parents[1]/'validation/ACADEMIC_WORD_PYTHON_COVERAGE.md').read_text()
    analyses=[p for p in analysis_catalog() if domain(p) is not None]
    assert len(analyses)==38
    assert all('| '+page+' |' in text for page in analyses)
