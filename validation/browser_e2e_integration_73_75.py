"""Real app upload -> SPSS controls -> calibrated Rasch -> downloads -> Wright."""
import argparse,json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from validation.browser_e2e_polytomous_dif import ROOT,Server,upload,nav,wait_idle,download


def run(output,app_python=None,browser_path=None):
    from playwright.sync_api import sync_playwright,expect
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    result=dict(status='FAIL',app='app.py',data='existing controlled Rasch fixture')
    page=None
    with (out/'streamlit.log').open('w') as log:
        try:
            with Server(log,python=app_python) as server,sync_playwright() as pw:
                browser=pw.chromium.launch(headless=True,executable_path=browser_path)
                page=browser.new_page(accept_downloads=True,viewport=dict(width=1440,height=1100))
                page.context.tracing.start(screenshots=True,snapshots=True,sources=True)
                try:
                    def click_text(label):
                        locator=page.get_by_text(label,exact=True)
                        locator.evaluate("el => el.scrollIntoView({block:'center'})")
                        locator.click()
                    def set_checkbox(label,value):
                        control=page.get_by_role('checkbox',name=label,exact=True)
                        if control.is_checked()!=value:click_text(label)
                        (expect(control).to_be_checked() if value else expect(control).not_to_be_checked())
                        wait_idle(page)
                    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                    upload(page,server.port,ROOT/'tests/fixtures/rasch/data_rasch_main.csv')
                    page.get_by_text('🔬 Ayrıntılı MTK Model Raporları',exact=True).first.click()
                    nav(page,'Python mirt — Rasch')
                    click_text('Çıktılar ve raporlama — SPSS tarzı seçimler')
                    names=['Model özeti, LogLik, AIC/BIC ve yakınsama','Madde parametreleri',
                        'Birey EAP yetenekleri ve posterior SH','Madde/kategori tepki eğrileri',
                        'Madde ve test bilgisi, koşullu SEM','Aynı veri üzerinde model karşılaştırması (ML AIC/BIC)',
                        'Madde uyumu: Infit, Outfit, S-X² (kütüphane API)']
                    for label in names:expect(page.get_by_role('checkbox',name=label,exact=True)).to_be_checked()
                    page.get_by_role('button',name='Python mirt ile hesapla',exact=True).click()
                    page.get_by_text('Kestirim yakınsadı; bilimsel sertifika anlamına gelmez.',exact=True).wait_for(timeout=180000)
                    wait_idle(page)
                    summary=pd.read_csv(download(page,'Model özeti — CSV',out))
                    assert summary['n_quadpts'].iloc[0]==21 and summary['max_iter'].iloc[0]==500 and summary['tol'].iloc[0]==1e-4
                    assert str(summary['Yakınsama'].iloc[0]).lower()=='true'
                    params=pd.read_csv(download(page,'Madde parametreleri — CSV',out))
                    people=pd.read_csv(download(page,'Birey yetenekleri — CSV',out))
                    person_xlsx=pd.read_excel(download(page,'Birey yetenekleri — Excel',out))
                    pd.testing.assert_frame_equal(people,person_xlsx,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
                    expected_root=ROOT/'validation/r_mirt_phase2/evidence'
                    ep=pd.read_csv(expected_root/'rasch_persons.csv');ei=pd.read_csv(expected_root/'rasch_items.csv')
                    assert len(people)==1000 and len(params)==10
                    np.testing.assert_allclose(people['Yetenek (θ)'],ep['θ'],rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(people['SH (θ)'],ep['Posterior SD'],rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(params.set_index('Madde').loc[ei.Madde,'b'],ei.b,rtol=1e-12,atol=1e-12)
                    page.get_by_text('Çıktılar ve raporlama — SPSS tarzı seçimler',exact=True).evaluate("el => el.scrollIntoView({block:'center'})")
                    page.screenshot(path=str(out/'default_rasch_spss.png'),full_page=True)
                    # Output controls must change actual visibility, preserving the fitted values.
                    set_checkbox('Madde/kategori tepki eğrileri',False)
                    expect(page.get_by_role('button',name='Tüm tepki eğrileri — CSV',exact=True)).to_have_count(0)
                    set_checkbox('Madde ve test bilgisi, koşullu SEM',False)
                    expect(page.get_by_role('button',name='Madde/test bilgisi ve SEM — CSV',exact=True)).to_have_count(0)
                    after=pd.read_csv(download(page,'Madde parametreleri — CSV',out))
                    pd.testing.assert_frame_equal(after,params)
                    set_checkbox('Madde/kategori tepki eğrileri',True)
                    set_checkbox('Madde ve test bilgisi, koşullu SEM',True)
                    click_text('Wright Map — Rasch (Python mirt)')
                    set_checkbox('Kütüphane Wright Map grafiğini göster',True)
                    page.locator('[data-testid=stPlotlyChart] .js-plotly-plot').wait_for(timeout=60000)
                    wp=pd.read_csv(download(page,'Wright persons — CSV',out))
                    wi=pd.read_csv(download(page,'Wright items — CSV',out))
                    np.testing.assert_allclose(wp['θ'],people['Yetenek (θ)'],rtol=0,atol=1e-14)
                    np.testing.assert_allclose(wp['Posterior SD'],people['SH (θ)'],rtol=0,atol=1e-14)
                    chart=page.locator('[data-testid=stPlotlyChart] .js-plotly-plot').evaluate("""g => ({
                        items:Array.from(g._fullData[1].y),names:Array.from(g._fullData[1].text),
                        counts:Array.from(g._fullData[0].x),axes:g._fullData.map(t=>t.yaxis||'y')
                    })""")
                    assert chart['axes']==['y','y'] and chart['names']==wi.Madde.tolist()
                    np.testing.assert_allclose(chart['items'],wi.b,rtol=1e-12,atol=1e-12)
                    assert -sum(chart['counts'])==1000
                    page.locator('[data-testid=stPlotlyChart]').evaluate("el => el.scrollIntoView({block:'center'})")
                    page.screenshot(path=str(out/'rasch_wright_app.png'),full_page=True)
                    click_text('Python mirt — sayısal çözücü')
                    expect(page.get_by_role('checkbox',name='Gelişmiş sayısal ayarları etkinleştir',exact=True)).not_to_be_checked()
                    set_checkbox('Gelişmiş sayısal ayarları etkinleştir',True)
                    for label,value in [('Maksimum EM iterasyonu','700'),('Mutlak EM amaç fonksiyonu toleransı','0.000001'),('Gauss–Hermite düğüm sayısı','31')]:
                        widget=page.get_by_role('spinbutton',name=label,exact=True)
                        widget.fill(value);widget.press('Tab')
                    expect(page.get_by_role('button',name='Model özeti — CSV',exact=True)).to_have_count(0)
                    page.get_by_role('button',name='Python mirt ile hesapla',exact=True).click()
                    page.get_by_text('Kestirim yakınsadı; bilimsel sertifika anlamına gelmez.',exact=True).wait_for(timeout=180000)
                    advanced=pd.read_csv(download(page,'Model özeti — CSV',out))
                    assert advanced['n_quadpts'].iloc[0]==31 and advanced['max_iter'].iloc[0]==700 and advanced['tol'].iloc[0]==1e-6
                    page.screenshot(path=str(out/'advanced_solver_app.png'),full_page=True)
                    tolerance=page.get_by_role('spinbutton',name='Mutlak EM amaç fonksiyonu toleransı',exact=True)
                    tolerance.fill('0');tolerance.press('Tab')
                    page.get_by_role('button',name='Python mirt ile hesapla',exact=True).click()
                    page.get_by_text('Kestirim yapılamadı: tol must be finite and positive',exact=True).wait_for(timeout=60000)
                    expect(page.get_by_role('button',name='Birey yetenekleri — CSV',exact=True)).to_have_count(0)
                    expect(page.locator('[data-testid=stPlotlyChart]')).to_have_count(0)
                    assert not errors,errors
                    page.screenshot(path=str(out/'invalid_solver_app.png'),full_page=True)
                    result.update(status='PASS',persons=1000,items=10,controls_verified=names,
                        solver_settings=dict(max_iter=700,tol=1e-6,n_quadpts=31),wright=chart,
                        parameter_values_unchanged_by_output_controls=True,invalid_settings_withheld=True)
                except Exception:
                    page.screenshot(path=str(out/'failure.png'),full_page=True)
                    raise
                finally:
                    page.context.tracing.stop(path=str(out/'trace.zip'))
                    browser.close()
        except Exception:
            result['error']=traceback.format_exc()
            if page is not None:
                try:page.screenshot(path=str(out/'failure.png'),full_page=True)
                except Exception:pass
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result['status']=='PASS'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--app-python');p.add_argument('--browser-path')
    args=p.parse_args()
    raise SystemExit(0 if run(args.output,args.app_python,args.browser_path) else 1)
