"""Frozen evidence/statistics only: no calibration, simulation or retries in CI."""
import gzip
import hashlib
import json
from pathlib import Path
import zipfile
import io
import numpy as np
from scipy.stats import binom
from validation.grm_fwer_sequential.statistics import summary,TAIL_ALPHA
from validation.grm_fwer_sequential.archive import record_stream

ROOT=Path('validation/grm_fwer_sequential/results')


def numerical_equal(actual,saved):
    if isinstance(actual,dict):
        assert actual.keys()==saved.keys()
        for key in actual:numerical_equal(actual[key],saved[key])
    elif isinstance(actual,list):
        assert len(actual)==len(saved)
        for a,b in zip(actual,saved):numerical_equal(a,b)
    elif isinstance(actual,float):
        np.testing.assert_allclose(actual,saved,rtol=1e-12,atol=1e-14)
    else:assert actual==saved


def test_canonical_evidence_identity_and_registered_sources():
    for path,digest in json.loads((ROOT/'SHA256.json').read_text()).items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    metadata=json.loads((ROOT/'metadata.json').read_text())
    for family in ('sources','archives'):
        for path,digest in metadata[family].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    assert metadata['historical_fits_reused']==0
    audit=json.loads((ROOT/'audit.json').read_text())
    assert audit['status']=='INDEPENDENT RAW AUDIT PASS'


def test_all_raw_decisions_reproduce_summary_and_simultaneous_cp():
    rows=[]
    with record_stream(ROOT) as stream:
        for line in stream:
            row=json.loads(line)
            rows.append({k:row[k] for k in ('seed','impact','valid','decisions')})
    actual=summary(rows);saved=json.loads((ROOT/'summary.json').read_text())
    numerical_equal(actual['conditions'],saved['conditions'])
    numerical_equal(actual['paired_operational_difference'],saved['paired_operational_difference'])
    assert len(rows)==json.loads((ROOT/'projection.json').read_text())['records']==saved['attempted']
    for c in actual['conditions'].values():
        n=c['n'];k=c['operational_events'];lo,hi=c['operational_cp']
        if k:assert abs(binom.sf(k-1,n,lo)-TAIL_ALPHA)<1e-10
        if k<n:assert abs(binom.cdf(k,n,hi)-TAIL_ALPHA)<1e-10
        if c['decision']=='SCOPED CONTROL (LIBERAL .075)':assert c['missing_outcome_robust_cp'][1]<=.075
        if c['decision']=='SCOPED INFLATION':assert c['missing_outcome_robust_cp'][0]>.05


def test_first_thirty_matrix_pairs_are_actual_saved_data():
    hashes={}
    with record_stream(ROOT) as stream:
        for line in stream:
            row=json.loads(line)
            if row['seed']<2026110130:hashes[f'{row["seed"]}_{row["impact"]}.npz']=row['data_sha256']
    with zipfile.ZipFile(ROOT/'matrices.zip') as archive:
        assert set(archive.namelist())==set(hashes) and len(hashes)==60
        for name in archive.namelist():
            with np.load(io.BytesIO(archive.read(name))) as f:
                assert hashlib.sha256(f['X'].tobytes()+f['group'].tobytes()).hexdigest()==hashes[name]
