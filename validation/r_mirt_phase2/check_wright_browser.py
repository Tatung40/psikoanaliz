"""Render already calibrated Rasch plot data; absolutely no model fit."""
import argparse,base64,json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.io as pio


def vector(value):
    """Plotly6 round-trips typed arrays as dtype/bdata objects in Python."""
    if isinstance(value,dict):
        if set(value)!={'dtype','bdata'}:raise ValueError('Unexpected Plotly vector encoding')
        dtype=np.dtype(value['dtype'])
        if dtype.kind not in 'fiu':raise ValueError('Nonnumeric Plotly vector')
        return np.frombuffer(base64.b64decode(value['bdata'],validate=True),dtype=dtype)
    result=np.asarray(value)
    if result.ndim!=1:raise ValueError('Plotly vector must be one-dimensional')
    return result


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    report=dict(status='FAIL',fits=0)
    try:
        from playwright.sync_api import sync_playwright
        root=Path(__file__).parent/'evidence'
        fig=pio.from_json((root/'rasch_wright.json').read_text())
        persons=pd.read_csv(root/'rasch_persons.csv');items=pd.read_csv(root/'rasch_items.csv');hist=pd.read_csv(root/'rasch_histogram.csv')
        theta=persons['θ'].to_numpy();edges=np.r_[hist.Alt,hist.Üst.iloc[-1]]
        counts,_=np.histogram(theta,bins=edges)
        np.testing.assert_array_equal(hist['Kişi sayısı'],counts)
        assert items.b.is_monotonic_increasing and len(persons)==1000 and len(items)==10
        np.testing.assert_allclose(vector(fig.data[1].y),items.b,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(vector(fig.data[0].y),hist.Merkez,rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(-vector(fig.data[0].x),counts)
        assert list(fig.data[1].text)==items.Madde.tolist()
        html=pio.to_html(fig,include_plotlyjs=True,full_html=True,div_id='wright')
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True)
            page=browser.new_page(viewport=dict(width=1200,height=750))
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.set_content(html,wait_until='load')
            page.wait_for_function("document.getElementById('wright')._fullLayout !== undefined")
            page.locator('#wright .scatterlayer .trace .point').first.wait_for()
            trace_script="""() => {let g=document.getElementById('wright');return {
                y:Array.from(g._fullData[1].y), names:Array.from(g._fullData[1].text), counts:Array.from(g._fullData[0].x),
                range:g._fullLayout.yaxis.range, axes:g._fullData.map(t=>t.yaxis||'y'),
                itemMarkers:g.querySelectorAll('.scatterlayer .trace .point').length,
                bars:g.querySelectorAll('.barlayer .point').length}}"""
            actual=page.evaluate(trace_script)
            assert not errors,errors
            assert actual['axes']==['y','y'] and actual['names']==items.Madde.tolist()
            np.testing.assert_allclose(actual['y'],items.b,rtol=1e-12,atol=1e-12)
            np.testing.assert_array_equal(-np.asarray(actual['counts']),counts)
            assert actual['itemMarkers']==len(items) and np.count_nonzero(counts)<=actual['bars']<=len(hist)
            assert actual['range'][0]<min(theta.min(),items.b.min()) and actual['range'][1]>max(theta.max(),items.b.max())
            page.screenshot(path=str(out/'rasch_wright.png'),full_page=True)
            # Real public LSAT6 is a separate actual fit; only aggregates saved.
            real=json.loads((root/'lsat6_summary.json').read_text())
            ri=pd.read_csv(root/'lsat6_items.csv');rh=pd.read_csv(root/'lsat6_histogram.csv')
            rf=pio.from_json((root/'lsat6_wright.json').read_text())
            assert real['status']=='PASS' and real['converged'] is True and real['fits']==1
            assert real['respondent_rows_saved'] is False and real['N']==1000 and len(ri)==5
            assert ri.b.is_monotonic_increasing and int(rh['Kişi sayısı'].sum())==real['N']
            np.testing.assert_allclose(vector(rf.data[1].y),ri.b,rtol=1e-12,atol=1e-12)
            np.testing.assert_array_equal(-vector(rf.data[0].x),rh['Kişi sayısı'])
            np.testing.assert_allclose(vector(rf.data[0].y),rh.Merkez,rtol=1e-12,atol=1e-12)
            page.set_content(pio.to_html(rf,include_plotlyjs=True,full_html=True,div_id='wright'),wait_until='load')
            page.wait_for_function("document.getElementById('wright')._fullLayout !== undefined")
            page.locator('#wright .scatterlayer .trace .point').first.wait_for()
            ra=page.evaluate(trace_script)
            assert not errors,errors
            assert ra['axes']==['y','y'] and ra['names']==ri.Madde.tolist() and ra['itemMarkers']==5
            np.testing.assert_allclose(ra['y'],ri.b,rtol=1e-12,atol=1e-12)
            np.testing.assert_array_equal(-np.asarray(ra['counts']),rh['Kişi sayısı'])
            assert np.count_nonzero(rh['Kişi sayısı'])<=ra['bars']<=len(rh)
            assert ra['range'][0]<min(real['theta_min'],ri.b.min()) and ra['range'][1]>max(real['theta_max'],ri.b.max())
            page.screenshot(path=str(out/'lsat6_wright.png'),full_page=True)
            report['real_LSAT6']=dict(persons=real['N'],items=len(ri),source=real['source'],data_sha256=real['data_sha256'],rendered=ra)
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
