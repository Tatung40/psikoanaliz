"""Canonical archived bytes vs regenerated GRM log coordinates.

Only one binary64 ULP is allowed in recomputed log coordinates. Data/file hashes,
optimizer settings, and scientific score/information/LR gates remain exact.
Historical audit source files and result archives are never edited.
"""
from contextlib import contextmanager
import gzip
import hashlib
import json
from unittest.mock import patch

import numpy as np


def float_bytes(values):
    return np.ascontiguousarray(values,dtype='<f8').tobytes()


def parameter_id(roles,params):
    return hashlib.sha256(json.dumps(tuple(roles)).encode()+float_bytes(params)).hexdigest()


def assert_reconstructed_parameters(canonical,recomputed,design,*,cold=False):
    a=np.asarray(canonical,dtype=np.float64);b=np.asarray(recomputed,dtype=np.float64)
    assert a.shape==b.shape==(design.size,)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    log_coordinates=np.zeros(design.size,dtype=bool)
    for j,(kind,group,item,c) in enumerate(design.labels):
        log_coordinates[j]=kind=='b' and (cold or c>0)
    if not cold:log_coordinates[-1]=True  # log(SD), reconstructed from saved SD
    # Slopes, means, fixed cold latent coordinates and copied thresholds stay exact.
    np.testing.assert_array_equal(a[~log_coordinates],b[~log_coordinates])
    np.testing.assert_array_max_ulp(a[log_coordinates],b[log_coordinates],maxulp=1)


def bind_archived_models(report,runtime_models,runtime_slots):
    """Verify immutable IDs from their ORIGINAL bytes; bind by role/slot semantics."""
    archived_slots=report['slots'];starts=report['stages'][0]['solutions']
    assert set(archived_slots)==set(runtime_slots)
    assert set(runtime_models)==set(runtime_slots.values())
    assert set(starts)==set(archived_slots.values())
    mapping={};reverse={};bound={}
    for slot,key in archived_slots.items():
        runtime_key=runtime_slots[slot];model=runtime_models[runtime_key];saved=starts[key]
        assert tuple(saved['roles'])==model['design'].roles
        assert parameter_id(saved['roles'],saved['initial_params'])==key
        assert_reconstructed_parameters(saved['initial_params'],model['params'],model['design'])
        assert mapping.setdefault(runtime_key,key)==key
        assert reverse.setdefault(key,runtime_key)==runtime_key
        bound[key]=model
    assert len(bound)==len(runtime_models)
    return bound,dict(archived_slots)


@contextmanager
def canonical_cold_audit(audit_module,output):
    """Read-only replay of the frozen auditor with validated canonical encodings.

The original independent frequency-count reconstruction ALWAYS executes first.
Only after the one-ULP semantic check may the original recorded serialization be
returned for its strict byte/hash checks. No statistical assertion is patched.
"""
    rows=[json.loads(line) for line in gzip.decompress((output/'records.jsonl.gz').read_bytes()).splitlines()]
    prior=audit_module.P.legacy_records();designs,_=audit_module.P.layouts();canonical={}
    def data_key(d,x):
        return tuple(d.roles),hashlib.sha256(np.ascontiguousarray(x,dtype='u1').tobytes()).hexdigest()
    for row in rows:
        x,g=audit_module.P.load_matrix(prior[(row['seed'],row['impact'])])
        for call in row['optimizer_calls']:
            d=designs[call['solution']]['design'];stored=np.asarray(call['x0'],dtype=np.float64)
            assert call['cold_verified']
            assert hashlib.sha256(float_bytes(stored)).hexdigest()==call['expected_cold_sha256']
            key=data_key(d,x)
            if key in canonical:np.testing.assert_array_equal(canonical[key],stored)
            canonical[key]=stored
    original=audit_module.independent_cold_start
    def reconstructed(d,x):
        calculated=original(d,x)
        stored=canonical[data_key(d,x)]
        assert_reconstructed_parameters(stored,calculated,d,cold=True)
        return stored.copy()
    with patch.object(audit_module,'independent_cold_start',reconstructed):
        yield
