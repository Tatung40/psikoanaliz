import numpy as np
import gzip,hashlib,json,pytest
from validation.grm_fwer_sequential.archive import compact,canonical_bytes,record_stream


def test_only_reproducible_matrix_cache_is_projected_without_filtering_results():
    row=dict(seed=1,valid=False,failure='gate',decisions={'p':None},solutions={
        'one':dict(params=[1,2],dense_information=dict(matrix=[[2.,1.],[1.,3.]],valid=True,eigenvalues=[1.,4.]))})
    new=compact(row)
    assert row['solutions']['one']['dense_information']['matrix']==[[2.,1.],[1.,3.]]
    info=new['solutions']['one']['dense_information']
    assert info['matrix_shape']==[2,2] and len(info['matrix_float64_sha256'])==64
    del info['matrix_shape'];del info['matrix_float64_sha256'];info['matrix']=row['solutions']['one']['dense_information']['matrix']
    assert new==row  # No estimates, failures or decisions are removed/rewritten.


def test_binary_shards_reassemble_exactly_and_corruption_fails_closed(tmp_path):
    payload=gzip.compress(b'{"seed":1,"valid":false}\n',mtime=0);parts=[]
    for i,b in enumerate((payload[:10],payload[10:])):
        p=tmp_path/f'part{i}';p.write_bytes(b)
        parts.append({'path':p.name,'sha256':hashlib.sha256(b).hexdigest()})
    (tmp_path/'records.compact.parts.json').write_text(json.dumps({'parts':parts,'canonical_sha256':hashlib.sha256(payload).hexdigest()}))
    assert canonical_bytes(tmp_path)==payload
    with record_stream(tmp_path) as f:assert f.read()==b'{"seed":1,"valid":false}\n'
    (tmp_path/'part1').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='part identity'):canonical_bytes(tmp_path)
