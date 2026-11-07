"""Replay actual archived evidence and offline exports; no new R fits."""
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

from validation.gtheory_advanced.archive import audit,ROOT
from validation.gtheory_advanced.verify import saved_case


def test_examples_and_no_mc_scope_preserve_old_estimators():
    import hashlib
    from validation.gtheory_advanced.ci_scope import permitted
    root=ROOT/'validation/gtheory_advanced/examples'
    manifest=json.loads((root/'manifest.json').read_text())
    assert len(manifest['files'])==6
    for name,meta in manifest['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==meta['sha256']
        assert len(pd.read_csv(root/name))==meta['rows']
    assert permitted(['core/generalizability_advanced.py','core/reproducible.py','validation/gtheory_advanced/evidence.json'])
    assert not permitted(['core/generalizability.py'])
    assert not permitted(['core/irt_binary_inference.py'])
    assert not permitted(['.github/workflows/r-4pl-engine.yml'])


def test_all_saved_native_references_and_scientific_failures_are_preserved():
    d=audit(ROOT/'validation/gtheory_advanced/evidence.json')
    native=d['comparisons'][:-1]
    assert len(native)==10 and sum(r['result']['converged'] for r in native)==8
    rejected=[r for r in native if not r['result']['converged']]
    assert len(rejected)==2
    for r in rejected:
        assert r['result']['design']=='p x (i:r)' and r['result']['singular']
        assert r['result']['scientific_fit_status']=='NOT CERTIFIED'
        assert r['result']['coefficients_status']=='NOT CERTIFIED / SAFETY REJECTION CONFIRMED'
        case=r['case'];files={f['path'][len(case)+1:]:f['content'] for f in d['files'] if f['path'].startswith(case+'/')}
        assert saved_case(files)==r['result']  # original two failed checks, without repeating fits
    assert d['original_ci_conclusion']=='failure' and d['original_failed_tests']==2
    assert len(d['profile_intervals'])==3


def test_actual_reml_report_exports_include_diagnostics_and_profile_intervals():
    from ui.generalizability_view import exports
    d=audit(ROOT/'validation/gtheory_advanced/evidence.json')
    record=next(r for r in d['comparisons'] if r['result'].get('design')=='p x i')
    prefix=record['case']+'/'
    files={f['path'][len(prefix):]:f['content'] for f in d['files'] if f['path'].startswith(prefix)}
    result={name:pd.read_csv(io.StringIO(files['python_'+name+'.csv']),index_col=0)
            for name in ('variance','g','errors','d','diagnostics','random_effects','profile_ci')}
    result.update(settings=json.loads(files['settings.json']),status='LIBRARY TESTED',method='R lme4 REML; saved actual fit, no new execution',
                  counts={'p':8,'i':4},anova=pd.DataFrame(),henderson=pd.DataFrame(),log=files['adapter.log'])
    profile_prefix=d['profile_test']+'/'
    profile_files={f['path'][len(profile_prefix):]:f['content'] for f in d['files'] if f['path'].startswith(profile_prefix)}
    profile_variance=pd.read_csv(io.StringIO(profile_files['variance.csv']),index_col=0)
    np.testing.assert_allclose(profile_variance.Variance,result['variance'].Variance,rtol=1e-8,atol=1e-8)
    result['profile_ci']=pd.DataFrame(d['profile_intervals']).drop(columns='Unnamed: 0',errors='ignore')
    result['settings']['profile']=True
    result['settings']['profile_source']=d['profile_test']+' (same fixture and estimates, actual native R profile)'
    tables,exported,report=exports(result,.8)
    assert {'diagnostics','random_effects','profile_ci'}<=set(tables)
    assert 'R lme4 REML' in report and exported['variance.png'].startswith(b'\x89PNG')
    workbook=pd.ExcelFile(io.BytesIO(exported['G_D.xlsx']))
    assert {'diagnostics','random_effects','profile_ci'}<=set(workbook.sheet_names)
    assert len(pd.read_excel(io.BytesIO(exported['G_D.xlsx']),sheet_name='profile_ci'))==3


def test_offline_python_bundle_includes_verified_reml_bridge_and_runs_multivariate(tmp_path):
    from core import generalizability_advanced as ga
    from core.reproducible import Recorder,python_code,bundle,runtime_hashes
    rec=Recorder('G-theory matrix MoM')
    rec.invoke(ga.multivariate,(ga.example('multivariate'),'p','i','Dimension','score'),{})
    manifest=rec.manifest();code=python_code(manifest);compile(code,'analysis.py','exec')
    assert 'scripts/gtheory_reml.R' not in manifest['runtime_sha256']
    with zipfile.ZipFile(io.BytesIO(bundle(manifest,code))) as z:
        assert 'runtime/scripts/gtheory_reml.R' not in z.namelist()
        z.extractall(tmp_path)
    done=subprocess.run([sys.executable,'analysis.py'],cwd=tmp_path,capture_output=True,text=True,timeout=60)
    assert done.returncode==0,done.stderr[-1500:]
    # Dependency packaging only, not a claimed fresh REML execution.
    calls=[{'module':'core.generalizability_advanced','function':'analyze','arguments':{}}]
    reml={**manifest,'calls':calls,'runtime_sha256':runtime_hashes(calls)}
    with zipfile.ZipFile(io.BytesIO(bundle(reml,python_code(reml)))) as z:
        assert 'runtime/scripts/gtheory_reml.R' in z.namelist()
