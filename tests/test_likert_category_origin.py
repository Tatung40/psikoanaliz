"""One fixed five-category dataset; real library fits, no new estimator."""
import numpy as np
import pandas as pd
import pytest
import mirt
from core.irt_b14 import prepare_polytomous,fit_b14
from core.irt_python_mirt_report import build_report
from core.irt_python_mirt_adapter import fit_python_mirt

@pytest.fixture(scope='module')
def zero():
    return pd.DataFrame(mirt.simdata(model='GRM',n_persons=300,n_items=6,n_categories=5,seed=94),
                        columns=[f'M{i}' for i in range(6)]).astype(int)


@pytest.mark.parametrize('model',['GRM','PCM','GPCM','NRM'])
def test_real_five_category_fits_preserve_labels_and_nominal_mapping(zero,model):
    raw=zero+1
    labels=('5','2','1','4','3') if model=='NRM' else None
    result=fit_b14(raw,model,categories=labels)
    assert result['result'].converged
    assert result['labels']==(labels if model=='NRM' else ('1','2','3','4','5'))
    expected=raw.replace({int(label):i for i,label in enumerate(labels)}).astype(int) if labels else zero
    pd.testing.assert_frame_equal(result['coded'],expected)
    assert np.isfinite(result['result'].log_likelihood)
    if model=='GRM':
        baseline=fit_b14(zero,'GRM')
        assert baseline['labels']==('0','1','2','3','4')
        pd.testing.assert_frame_equal(result['result'].coef(),baseline['result'].coef())
        assert result['result'].log_likelihood==baseline['result'].log_likelihood


@pytest.mark.parametrize('model',['GRM','PCM','GPCM'])
def test_missing_category_is_not_squeezed_or_fabricated(zero,model):
    with pytest.raises(ValueError,match='Eksik:'):
        prepare_polytomous((zero+1).replace(3,2),model)
    one_item=zero+1;one_item.iloc[:,0]=one_item.iloc[:,0].replace(3,2)
    with pytest.raises(ValueError,match='Her maddede'):
        fit_b14(one_item,model)
    with pytest.raises(ValueError,match='0 veya 1'):
        prepare_polytomous(zero+2,model)


def test_nominal_missing_declared_label_stays_an_error(zero):
    with pytest.raises(ValueError,match='Eksik:'):
        prepare_polytomous((zero+1).replace(3,2),'NRM',categories=('5','2','1','4','3'))


def test_ui_infers_five_not_six_and_retains_original_curve_labels(zero,monkeypatch):
    from streamlit.testing.v1 import AppTest
    from pathlib import Path
    import ui.irt_python_library_view as view
    calls=[]
    def actual(data,**kwargs):
        calls.append(np.asarray(data).copy())
        return fit_python_mirt(data,**kwargs)
    monkeypatch.setattr(view,'fit_python_mirt',actual)
    at=AppTest.from_file(str(Path('app.py').resolve()),default_timeout=120)
    raw=zero+1
    for k,v in dict(raw_df=raw,scored_df=None,item_cols=list(raw),id_col=None,group_col=None,is_dichotomous=False,nav_page='MTK: Python mirt GRM').items():at.session_state[k]=v
    at.run()
    assert not at.exception and at.number_input(key='python_mirt_categories_GRM').value==5
    at.button(key='python_mirt_start_GRM').click().run()
    assert not at.exception and any('Kestirim yakınsadı' in e.value for e in at.success)
    np.testing.assert_array_equal(calls[0],zero)
    outcome=at.session_state['python_mirt_fit_GRM']['outcome']
    report=build_report(outcome,model='GRM',items=zero.columns,labels=('1','2','3','4','5'))
    assert set(report.curves.Kategori)==set('12345')
    at.session_state['raw_df']=zero;at.run()
    assert not at.exception and not at.dataframe and not at.get('download_button') and len(calls)==1
