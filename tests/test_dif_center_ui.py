"""Actual Streamlit routes and fail-closed cached reports; no model fits here."""
from pathlib import Path
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from tests.test_dif_center import fixture_data,request

APP=str(Path('app.py').resolve())


def app():
    data=fixture_data();items=request(data)['items'];data=data[['Grup']+items]
    at=AppTest.from_file(APP,default_timeout=60)
    for key,value in dict(raw_df=data,scored_df=data,item_cols=items,id_col=None,group_col='Grup',
                          nav_page='MTK: DIF Analiz Merkezi',is_dichotomous=True).items():
        at.session_state[key]=value
    return at.run()


def test_actual_center_flow_downloads_and_unchanged_result_retention():
    at=app();assert not at.exception
    at.button(key='dc_run').click().run()
    assert not at.exception and not at.error
    assert len(at.session_state['dif_center_result']['result']['table'])==6
    assert len(at.get('download_button'))>=5
    key=at.session_state['dif_center_result']['key']
    at.run()
    assert at.session_state['dif_center_result']['key']==key
    assert len(at.get('download_button'))>=5


@pytest.mark.parametrize('change',['values','alpha','correction','anchors','hypothesis','method','sib_correction'])
def test_center_stale_tables_plots_and_exports_are_suppressed(change):
    at=app();at.button(key='dc_run').click().run();assert not at.exception
    if change=='values':
        data=at.session_state['raw_df'].copy();data.iloc[0,1]=1-data.iloc[0,1]
        at.session_state['raw_df']=data
    elif change=='alpha':at.number_input(key='dc_alpha').set_value(.1)
    elif change=='correction':at.selectbox(key='dc_correction').set_value('BH')
    elif change=='anchors':at.multiselect(key='dc_anchors').set_value(['M1','M2','M3'])
    elif change=='hypothesis':at.selectbox(key='dc_lr_test').set_value('uniform')
    elif change=='method':at.multiselect(key='dc_methods').set_value(['SIBTEST'])
    else:at.checkbox(key='dc_sib_correction').uncheck()
    at.run();assert not at.exception
    assert 'dif_center_result' not in at.session_state
    assert not at.get('download_button')
    assert not any('Statistic' in d.value for d in at.dataframe)


@pytest.mark.parametrize('route',['IRT-LR / Raju alanı','Çapa arındırması / ordinal gözlenen puan',
                                  'GRM formel DIF','PCM / GPCM / RSM / NRM formel DIF'])
def test_existing_routes_open_without_estimation(route):
    at=app();at.radio(key='dc_route').set_value(route).run()
    assert not at.exception
