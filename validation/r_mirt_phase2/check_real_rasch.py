"""One public LSAT6 Rasch fit; reuse existing source and APIs, retain aggregates only."""
import io,json,hashlib,platform,importlib.metadata,traceback
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from mirt.estimation.quadrature import GaussHermiteQuadrature
from scripts.real_lsat6_ctt_pilot import URL,EXPECTED_N,EXPECTED_P
from core.irt_python_mirt_adapter import fit_python_mirt
from core.irt_rasch_wright import rasch_wright_tables,wright_figure,wright_exports


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    report=dict(status='FAIL',source=URL,python=platform.python_version(),mirt=importlib.metadata.version('mirt'),fits=0)
    try:
        with urlopen(URL,timeout=25) as response:content=response.read()
        assert len(content)<100_000
        raw=pd.read_csv(io.BytesIO(content));x=raw[[f'Q{i}' for i in range(1,6)]]
        assert len(x)==EXPECTED_N and x.notna().all().all() and np.isin(x,[0,1]).all()
        np.testing.assert_allclose(x.mean(),EXPECTED_P,rtol=0,atol=1e-12)
        report.update(data_sha256=hashlib.sha256(content).hexdigest(),N=len(x),items=x.shape[1])
        report['fits']=1
        outcome=fit_python_mirt(x,model='Rasch')
        tables=rasch_wright_tables(outcome,x,item_names=x.columns,identifiers=raw['rownames'])
        persons,items,h=tables['persons'],tables['items'],tables['histogram']
        # Numerical validation only: independent posterior moments from actual API P.
        q=GaussHermiteQuadrature(n_points=61,n_dimensions=1);P=outcome.fit.model.probability(q.nodes)
        lp=np.log(q.weights)[None,:]+sum(np.log(np.where(x.iloc[:,i].to_numpy()[:,None]==1,P[:,i][None,:],1-P[:,i][None,:])) for i in range(x.shape[1]))
        post=np.exp(lp-logsumexp(lp,axis=1)[:,None]);theta=post@q.nodes[:,0]
        sd=np.sqrt((post*(q.nodes[:,0][None,:]-theta[:,None])**2).sum(1))
        np.testing.assert_allclose(persons['θ'],theta,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(persons['Posterior SD'],sd,rtol=1e-12,atol=1e-12)
        # b is the physical location: P(theta=b)=.5 at fixed a=1 (D=1).
        original=outcome.fit.coef().difficulty.to_numpy()
        np.testing.assert_allclose(np.diag(outcome.fit.model.probability(original[:,None])),.5,rtol=0,atol=1e-14)
        fig=wright_figure(tables);bars,points=fig.data
        np.testing.assert_array_equal(points.y,items.b)
        np.testing.assert_array_equal(bars.customdata,h['Kişi sayısı'])
        assert int(h['Kişi sayısı'].sum())==len(x)
        csv,excel=wright_exports(tables)
        for key,frame in tables.items():
            pd.testing.assert_frame_equal(pd.read_csv(io.BytesIO(csv[key])),frame,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
            pd.testing.assert_frame_equal(pd.read_excel(io.BytesIO(excel),sheet_name=key),frame,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
        # Only aggregate plot/histogram and item estimates leave this process.
        (out/'lsat6_wright.json').write_text(fig.to_json())
        items.to_csv(out/'lsat6_items.csv',index=False);h.to_csv(out/'lsat6_histogram.csv',index=False)
        report.update(status='PASS',converged=bool(outcome.fit.converged),iterations=outcome.fit.n_iterations,
            loglik=float(outcome.fit.log_likelihood),solver=outcome.solver_settings,
            max_theta_identity_error=float(abs(theta-persons['θ']).max()),
            max_SD_identity_error=float(abs(sd-persons['Posterior SD']).max()),
            theta_min=float(theta.min()),theta_max=float(theta.max()),histogram_count=len(x),item_order=items.Madde.tolist(),
            extreme_zero_count=int((x.sum(axis=1)==0).sum()),extreme_full_count=int((x.sum(axis=1)==x.shape[1]).sum()),
            respondent_rows_saved=False)
    except Exception:report['error']=traceback.format_exc()
    (out/'lsat6_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report['status']=='PASS'


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    raise SystemExit(0 if main(p.parse_args().output) else 1)
