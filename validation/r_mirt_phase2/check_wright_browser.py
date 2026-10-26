"""Render already calibrated Rasch plot data; absolutely no model fit."""
import argparse,json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.io as pio
from playwright.sync_api import sync_playwright

def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    report=dict(status='FAIL',fits=0)
    try:
        root=Path(__file__).parent/'evidence'
        fig=pio.from_json((root/'rasch_wright.json').read_text())
        persons=pd.read_csv(root/'rasch_persons.csv');items=pd.read_csv(root/'rasch_items.csv');hist=pd.read_csv(root/'rasch_histogram.csv')
        theta=persons['θ'].to_numpy();edges=np.r_[hist.Alt,hist.Üst.iloc[-1]]
        counts,_=np.histogram(theta,bins=edges)
        np.testing.assert_array_equal(hist['Kişi sayısı'],counts)
        assert items.b.is_monotonic_increasing and len(persons)==1000 and len(items)==10
        np.testing.assert_allclose(fig.data[1].y,items.b,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(fig.data[0].y,hist.Merkez,rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(-np.asarray(fig.data[0].x),counts)
        assert list(fig.data[1].text)==items.Madde.tolist()
        html=pio.to_html(fig,include_plotlyjs=True,full_html=True,div_id='wright')
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True)
            page=browser.new_page(viewport=dict(width=1200,height=750))
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.set_content(html,wait_until='load')
            page.wait_for_function("document.getElementById('wright')._fullLayout !== undefined")
            page.locator('#wright .scatterlayer .trace .point').first.wait_for()
            actual=page.evaluate("""() => {let g=document.getElementById('wright');return {
                y:Array.from(g._fullData[1].y), names:Array.from(g._fullData[1].text), counts:Array.from(g._fullData[0].x),
                range:g._fullLayout.yaxis.range, axes:g._fullData.map(t=>t.yaxis||'y'),
                itemMarkers:g.querySelectorAll('.scatterlayer .trace .point').length,
                bars:g.querySelectorAll('.barlayer .point').length}}""")
            assert not errors,errors
            assert actual['axes']==['y','y'] and actual['names']==items.Madde.tolist()
            np.testing.assert_allclose(actual['y'],items.b,rtol=1e-12,atol=1e-12)
            np.testing.assert_array_equal(-np.asarray(actual['counts']),counts)
            assert actual['itemMarkers']==len(items) and np.count_nonzero(counts)<=actual['bars']<=len(hist)
            assert actual['range'][0]<min(theta.min(),items.b.min()) and actual['range'][1]>max(theta.max(),items.b.max())
            page.screenshot(path=str(out/'rasch_wright.png'),full_page=True)
            browser.close()
        report.update(status='PASS',persons=len(persons),items=len(items),histogram_count=int(counts.sum()),rendered=actual)
    except Exception:
        report['error']=traceback.format_exc()
    (out/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report['status']=='PASS'

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    raise SystemExit(0 if main(parser.parse_args().output) else 1)
