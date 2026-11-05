"""Actual UI output containers and names; no estimator or inference changes."""
import io
import re
from pathlib import Path
from docx import Document
import pytest
from streamlit.testing.v1 import AppTest
from ui.result_panels import docx_download

ROOT=Path(__file__).resolve().parents[1]


def walk(node,parents=()):
    yield node,parents
    for child in getattr(node,'children',{}).values():yield from walk(child,parents+(node,))


def assert_result_panels(at):
    outputs={id(x) for x in at.dataframe}
    outputs.update(id(x) for name in ('plotly_chart','imgs','arrow_vega_lite_chart','vega_lite_chart') for x in at.get(name))
    for node,parents in walk(at._tree):
        if id(node) not in outputs:continue
        # Editable model matrices are analysis inputs, not result tables.
        if getattr(getattr(node,'proto',None),'editing_mode',0):continue
        expanders=[p for p in parents if getattr(p,'type','')=='expander']
        assert len(expanders)==1,(getattr(node,'type',''),[p.label for p in expanders])
        assert not expanders[0].proto.expanded,expanders[0].label


@pytest.mark.parametrize('model',['Rasch','1PL','2PL','3PL','4PL','GRM','PCM','GPCM','RSM','NRM'])
def test_contextual_model_labels_and_safe_filenames(model):
    for page in ('MTK: '+model,'MTK: Python mirt '+model):
        label,filename=docx_download(page,{})
        assert label==model+' Akademik Rapor'
        assert filename==model.lower()+'_akademik_rapor.docx'
    label,filename=docx_download('MTK: Çok Kategorili B14',{'b14_model':model})
    assert label==model+' Akademik Rapor'


def test_all_routes_have_specific_names():
    from core.example_datasets import analysis_catalog
    for page in analysis_catalog():
        label,name=docx_download(page,{})
        assert label.endswith(' Akademik Rapor') and label!='Akademik Rapor'
        assert re.fullmatch(r'[a-z0-9_]+_akademik_rapor\.docx',name)
        assert 'Word indir' not in label
    assert docx_download('MTK: Python mirt DIF',{'pmd_model':'2PL'})[0]=='2PL DIF Akademik Rapor'
    assert docx_download('Çok Boyutlu MTK (MIRT)',{'md_model':'Bifactor'})[0]=='Bifactor MIRT Akademik Rapor'


def test_tables_charts_controls_alerts_and_downloads_keep_their_actual_content():
    script='''
import pandas as pd, streamlit as st
import plotly.graph_objects as go
from core.reproducible import finish
from ui.reproducible_capture import begin
from ui.academic_word import begin as word_begin,finish as word_finish
from core.descriptives import describe_items
x=pd.DataFrame({'İ1':[1.,2.,3.,4.],'Ö2':[4.,3.,2.,1.]})
begin('3B. Betimsel İstatistikler',globals());word_begin('3B. Betimsel İstatistikler',x,x,list(x))
st.subheader('Madde betimlemeleri')
st.number_input('Analiz ayarı',value=1,key='input');st.button('Hesapla',key='run')
table=describe_items(x);st.columns(2)[0].dataframe(table)
with st.expander('Mevcut tanı paneli',expanded=False):
 st.dataframe(table)
 st.warning('Kritik yakınsama uyarısı')
 st.download_button('CSV indir',b'exact-csv-bytes','original.csv',key='csv')
 st.plotly_chart(go.Figure(go.Scatter(x=[1,2],y=[.2,.8])))
st.bar_chart(x)
word_finish(finish())
'''
    at=AppTest.from_string(script,default_timeout=30).run()
    assert not at.exception,[e.value for e in at.exception]
    assert len(at.dataframe)==2 and len(at.get('plotly_chart'))==1
    assert_result_panels(at)
    protected={id(x) for x in list(at.warning)+list(at.button)+list(at.number_input)+list(at.get('download_button'))}
    for node,parents in walk(at._tree):
        if id(node) in protected:assert not any(getattr(p,'type','')=='expander' for p in parents)
    assert [x.label for x in at.get('download_button')]==['CSV indir','Betimsel İstatistikler Akademik Rapor']
    raw=at.session_state['_academic_history']['3B. Betimsel İstatistikler']['docx']
    doc=Document(io.BytesIO(raw));text='\n'.join(p.text for p in doc.paragraphs)
    assert 'Kritik yakınsama uyarısı' in text
    assert len(doc.tables)==2 and len(doc.inline_shapes)==1
    before=[p.text for p in doc.paragraphs];at.run();assert not at.exception
    assert_result_panels(at)
    after=Document(io.BytesIO(at.session_state['_academic_history']['3B. Betimsel İstatistikler']['docx']))
    assert [p.text for p in after.paragraphs]==before
