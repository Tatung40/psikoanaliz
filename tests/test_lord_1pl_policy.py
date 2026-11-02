"""1PL Lord inference closure using frozen actual fits; zero recalibrations."""
import io
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from core.dif_lord_policy import protect_result,STATUS,WARNING,DESCRIPTIVE_NOTE
from core.dif_center import observed,LIMITATIONS
from reports.dif_center_report import downloads,figure

ROOT=Path('validation/dif_wald_lord/results')


def saved_result(case='dif1pl_uniform'):
    root=ROOT/case;settings=json.loads((root/'settings.json').read_text())
    lord=pd.read_csv(root/'python_lord.csv',dtype={'Reason':'string'}).fillna({'Reason':''})
    wald=pd.read_csv(root/'python_wald.csv',dtype={'Reason':'string'}).fillna({'Reason':''})
    table=pd.concat([lord,wald],ignore_index=True)
    table['p_adj']=table.p;table['q']=table.p
    return dict(table=table,metadata={'Parametric_model':settings['model'],'Alpha':.05,'Limitations':LIMITATIONS})


@pytest.mark.parametrize('sigma',[.5,1.,1.000001,2.])
def test_unverified_scale_always_suppresses_1pl_lord_p_q_without_changing_numerics(sigma):
    raw=saved_result();before=raw['table'].copy(deep=True)
    raw['metadata']['Wald_sigma_F_diagnostic']=sigma
    result=protect_result(raw);lord=result['table'].Method.eq('Lord χ²')
    assert result['table'].loc[lord,['p','p_adj','q']].isna().all().all()
    assert (result['table'].loc[lord,'Status']==STATUS).all()
    assert result['table'].loc[lord,'Reason'].str.contains(WARNING,regex=False).all()
    pd.testing.assert_frame_equal(result['table'].loc[:,['Statistic','Delta_b','SE_b']],before[['Statistic','Delta_b','SE_b']])
    pd.testing.assert_frame_equal(result['table'].loc[~lord],before.loc[~lord])
    pd.testing.assert_frame_equal(raw['table'],before)  # Archived raw numerics are immutable.
    pd.testing.assert_frame_equal(protect_result(result)['table'],result['table'])


def test_direct_legacy_exports_cannot_leak_p_q_and_show_identical_status():
    raw=saved_result()
    for name in ('p_value','p_value_adjusted','p_adjusted','q_value','q (BH)','p (BH-FDR)'):
        raw['table'][name]=raw['table'].p
    payload=downloads(raw)
    frames=[pd.read_csv(io.BytesIO(payload['csv'])),pd.read_excel(io.BytesIO(payload['xlsx']),sheet_name='DIF')]
    with zipfile.ZipFile(io.BytesIO(payload['zip'])) as z:
        frames.append(pd.read_csv(io.BytesIO(z.read('DIF.csv'))))
        frames.append(pd.read_excel(io.BytesIO(z.read('report.xlsx')),sheet_name='DIF'))
        assert WARNING in z.read('report.md').decode()
    for table in frames:
        lord=table.Method.eq('Lord χ²')
        aliases=['p','p_adj','q','p_value','p_value_adjusted','p_adjusted','q_value','q (BH)','p (BH-FDR)']
        assert table.loc[lord,aliases].isna().all().all()
        assert (table.loc[lord,'Status']==STATUS).all()
        assert np.isfinite(table.loc[lord,'Statistic']).all()
    assert WARNING in payload['report'].decode() and DESCRIPTIVE_NOTE in payload['report'].decode()
    fig=figure(raw)
    line=next(line for line in fig.axes[0].lines if line.get_label()=='Lord χ²')
    assert np.isnan(line.get_ydata()).all()
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_2pl_lord_and_all_wald_rows_are_unchanged():
    raw=saved_result('dif2pl_mixed')
    assert protect_result(raw) is raw
    only_wald=saved_result();only_wald['table']=only_wald['table'].loc[lambda d:d.Method=='Wald DIF']
    assert protect_result(only_wald) is only_wald


def install_saved_engines(monkeypatch):
    root=ROOT/'dif1pl_uniform';cfg=json.loads((root/'settings.json').read_text())
    for name,target in [('lord','core.dif_lord.lord_dif'),('wald','core.irt_dif.wald_dif')]:
        result=dict(table=pd.read_csv(root/f'python_{name}.csv',dtype={'Reason':'string'}).fillna({'Reason':''}),
                    covariance=pd.read_csv(root/f'python_{name}_cov.csv'))
        monkeypatch.setattr(target,lambda *a,_result=result,**kw:_result)
    data=pd.read_csv('tests/fixtures/irt_b10/data_dif1pl_uniform.csv').dropna()
    return data,cfg


def test_production_center_closes_only_1pl_lord_and_preserves_wald(monkeypatch):
    data,cfg=install_saved_engines(monkeypatch)
    result=observed(data,items=cfg['items'],group='Grup',reference='R',focal='F',anchors=cfg['anchors'],
        studied=cfg['studied'],methods=['Lord χ²','Wald DIF'],model='1PL')
    lord=result['table'].Method.eq('Lord χ²')
    assert result['table'].loc[lord,['p','p_adj']].isna().all().all()
    assert (result['table'].loc[lord,'Status']==STATUS).all()
    assert result['table'].loc[lord,'Decision'].str.contains('ÇIKARIM KAPALI').all()
    assert result['table'].loc[~lord,'p'].notna().all()
    assert result['metadata']['Lord_warning']==WARNING


def test_actual_ui_uses_same_descriptive_status_and_warning(monkeypatch):
    from streamlit.testing.v1 import AppTest
    data,cfg=install_saved_engines(monkeypatch)
    at=AppTest.from_file(str(Path('app.py').resolve()),default_timeout=60)
    values=dict(raw_df=data,scored_df=data,item_cols=cfg['items'],id_col=None,group_col='Grup',
        is_dichotomous=True,nav_page='MTK: DIF Analiz Merkezi',dc_methods=['Lord χ²','Wald DIF'],
        dc_parametric_model='1PL',dc_anchors=cfg['anchors'],dc_studied=cfg['studied'])
    for key,value in values.items():at.session_state[key]=value
    at.run();at.button(key='dc_run').click().run()
    assert not at.exception and not at.error
    assert any(w.value==WARNING for w in at.warning)
    table=next(d.value for d in at.dataframe if {'Method','Statistic','p_adj'}<=set(d.value))
    lord=table.Method.eq('Lord χ²')
    assert table.loc[lord,['p','p_adj']].isna().all().all()
    assert (table.loc[lord,'Status']==STATUS).all()
    assert table.loc[~lord,'p'].notna().all()
