"""No-fit audit of original saved R/Python outputs, including failed fits."""
import io
import json
import re

import numpy as np
import pandas as pd


def saved_case(files):
    def frame(name):return pd.read_csv(io.StringIO(files[name]))
    settings=json.loads(files['settings.json']);diag=settings['convergence'];scale=settings['score_scale']
    actual=frame('python_variance.csv').set_index('source')
    reference=frame('reference/variance.csv')
    key=lambda s:frozenset(s.split(':'))
    expected={key(row.source):row.Variance*scale**2 for row in reference.itertuples()}
    difference={name:float(row.Variance-expected[key(name)]) for name,row in actual.iterrows()}
    tol=1e-8*max(1,max(expected.values()))
    assert max(abs(v) for v in difference.values())<=tol
    direct=frame('reference/fit.csv').iloc[0]
    assert diag['converged']==(int(direct.convergence_messages)==0)
    assert diag['singular']==bool(direct.singular)
    assert abs(diag['standardized_logLik']-direct.logLik)<1e-8
    assert abs(diag['logLik']-(direct.logLik-(diag['observations']-1)*np.log(scale)))<1e-8
    eligible=diag['converged'] and not diag['singular'] and diag['hessian_pd']
    if not eligible:
        assert frame('python_g.csv')[['rho^2','phi^2']].isna().all().all()
        assert frame('python_d.csv').empty
        coefficient_status='NOT CERTIFIED / SAFETY REJECTION CONFIRMED'
    else:
        counts={f:int(values[0]) for f,values in settings['scenarios'].items()}
        for row in frame('reference/coefficients.csv').itertuples():
            target={f:n if f=='p' else n*row.multiplier for f,n in counts.items()}
            U=float(actual.loc['p','Variance']);R=A=0.
            for name,value in actual.Variance.items():
                if name=='p':continue
                facets=set(re.findall('[pir]',name));part=value/np.prod([target[f] for f in facets if f!='p'])
                A+=part
                if 'p' in facets:R+=part
            np.testing.assert_allclose([U,R,A],[row.universe*scale**2,row.relative_error*scale**2,row.absolute_error*scale**2],rtol=1e-8,atol=tol)
            np.testing.assert_allclose([U/(U+R),U/(U+A)],[row.rho,row.phi],rtol=0,atol=1e-8)
        coefficient_status='NUMERICALLY VALIDATED: balanced target G/D, SEM and errors'
    return {'status':'NUMERICALLY VALIDATED' if eligible else 'SAFETY REJECTION CONFIRMED',
            'scientific_fit_status':'CONVERGED / NONSINGULAR' if eligible else 'NOT CERTIFIED',
            'design':settings['design'],'observations':diag['observations'],
            'variance_differences':difference,'converged':diag['converged'],'singular':diag['singular'],
            'hessian_pd':diag['hessian_pd'],'coefficients_status':coefficient_status,
            'boundary_components':diag['boundary_components'],'versions':settings['versions'],
            'fixture_sha256':settings['input_sha256']}
