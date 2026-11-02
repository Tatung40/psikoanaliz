"""No-fit regressions for current-input reporting and zero-information SEM."""
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from core.irt_b14_persons_info import information_table
from ui.irt_rsm_view import _fingerprint
from core.irt_binary_usage_policy import usage_status


@pytest.mark.parametrize('model,estimation', [('3PL', 'ML'), ('3PL', 'MAP'), ('4PL', 'MAP')])
def test_unstudied_design_is_not_reported_as_supported(model, estimation):
    for n, k in [(10001, 12), (10000, 13), (100000, 50)]:
        status, message = usage_status(model, estimation, n, k)
        assert status == 'caution'
        assert 'doğrudan incelenmiş' in message


RSM_APP = '''
import streamlit as st
from unittest.mock import patch
from ui import irt_rsm_view as view

def saved_analysis(*args):
    st.write('SAVED_ANALYSIS_RENDERED')
    st.stop()

with patch.object(view, '_analyse', saved_analysis):
    view.render(st.session_state.data, list(st.session_state.data.columns))
'''


def rsm_page():
    frame = pd.DataFrame({'I1': [0, 1, 2], 'I2': [2, 1, 0], 'I3': [1, 0, 2]})
    at = AppTest.from_string(RSM_APP)
    at.session_state['data'] = frame
    at.run()
    at.session_state['rsm_request'] = dict(cols=list(frame.columns), ident='rasch',
                                          points=41, lr=True, fp=_fingerprint(frame))
    return at


@pytest.mark.parametrize('change', ['values', 'items', 'identification', 'quadrature', 'comparison'])
def test_rsm_stale_results_and_downloads_are_suppressed(change):
    at = rsm_page()
    if change == 'values':
        frame = at.session_state['data'].copy()
        frame.iloc[0, 0] = 2  # Same shape/columns; actual values changed.
        at.session_state['data'] = frame
    elif change == 'items':
        at.multiselect(key='rsm_items').set_value(['I1', 'I2'])
    elif change == 'identification':
        at.radio(key='rsm_ident').set_value('unit_variance')
    elif change == 'quadrature':
        at.select_slider(key='rsm_points').set_value(61)
    else:
        at.checkbox(key='rsm_lr').uncheck()
    at.run()
    assert not at.exception
    assert 'rsm_request' not in at.session_state
    assert not any(x.value == 'SAVED_ANALYSIS_RENDERED' for x in at.markdown)
    assert not at.dataframe and not at.get('download_button') and not at.get('plotly_chart')


def test_rsm_unchanged_inputs_retain_saved_analysis():
    at = rsm_page().run()
    assert not at.exception
    assert 'rsm_request' in at.session_state
    assert any(x.value == 'SAVED_ANALYSIS_RENDERED' for x in at.markdown)


def test_zero_information_has_no_finite_sem():
    class Model:
        def information(self, theta, item_idx):
            return np.array([0., 2.])
    result = {'result': type('Fit', (), {'model': Model()})(), 'items': ['I1', 'I2']}
    table = information_table(result, np.array([-1., 1.]))
    assert table['Test bilgisi'].tolist() == [0., 4.]
    assert np.isnan(table['SEM(θ)'].iloc[0])
    assert table['SEM(θ)'].iloc[1] == .5


@pytest.mark.parametrize('model', ['3PL', '4PL', '1PL', 'GRM', 'PCM', 'GPCM'])
def test_itemfit_ui_and_export_policy(model):
    import io
    from types import SimpleNamespace
    from unittest.mock import patch
    from core.irt_library_diagnostics import library_itemfit
    source = pd.DataFrame({'infit': [1.], 'outfit': [1.], 'S_X2': [3.],
                           'df': [2], 'p_value': [.01], 'p_value_adjusted': [.02]})
    fit = SimpleNamespace(converged=True, model=object())
    with patch('mirt.itemfit', return_value=source) as api:
        table = library_itemfit(fit, [[0, 1]], model=model)
    statistics = api.call_args.kwargs['statistics']
    assert ('S_X2' in statistics) == (model in ('1PL', '3PL', '4PL'))
    assert table[['p_value', 'p_value_adjusted']].isna().all().all()
    assert table.S_X2.tolist() == [3.]  # Descriptive statistic is preserved.
    assert source.p_value.iloc[0] == .01  # Raw engine output is unaltered.
    assert 'NOT CERTIFIED' in table['Bilimsel durum'].iloc[0]
    assert pd.read_csv(io.StringIO(table.to_csv(index=False))).p_value.isna().all()
    output = io.BytesIO()
    table.to_excel(output, index=False)
    assert pd.read_excel(io.BytesIO(output.getvalue())).p_value.isna().all()


def test_itemfit_preserves_reference_checked_2pl_and_rejects_invalid_fits():
    from types import SimpleNamespace
    from unittest.mock import patch
    from core.irt_library_diagnostics import library_itemfit
    with patch('mirt.itemfit', return_value=pd.DataFrame({'p_value': [.2]})) as api:
        fit = SimpleNamespace(converged=True)
        assert library_itemfit(fit, [[0, 1]], model='2PL').p_value.iloc[0] == .2
        api.reset_mock()
        for model, converged in [('NRM', True), ('RSM', True), ('2PL', False)]:
            with pytest.raises(ValueError):
                library_itemfit(SimpleNamespace(converged=converged), [[0, 1]], model=model)
        api.assert_not_called()
