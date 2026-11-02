"""Drop only reproducible dense-matrix caches; retain every scientific result."""
import copy
import gzip
import hashlib
import json
import io
import numpy as np
from validation.grm_fwer_sequential.runner import HERE
from validation.gh_reoptimization.experiment import save


def canonical_bytes(output=HERE/'results'):
    manifest=output/'records.compact.parts.json'
    if not manifest.exists():
        path=output/('records.compact.jsonl.gz' if (output/'records.compact.jsonl.gz').exists() else 'records.jsonl.gz')
        return path.read_bytes()
    spec=json.loads(manifest.read_text());pieces=[]
    for part in spec['parts']:
        value=(output/part['path']).read_bytes()
        if hashlib.sha256(value).hexdigest()!=part['sha256']:raise ValueError('Archive part identity')
        pieces.append(value)
    value=b''.join(pieces)
    if hashlib.sha256(value).hexdigest()!=spec['canonical_sha256']:raise ValueError('Reassembled archive identity')
    return value


def record_stream(output=HERE/'results'):
    return gzip.GzipFile(fileobj=io.BytesIO(canonical_bytes(output)),mode='rb')


def compact(row):
    result=copy.deepcopy(row)
    for solution in result['solutions'].values():
        info=solution.get('dense_information',{})
        if 'matrix' in info:
            matrix=np.asarray(info.pop('matrix'),dtype='<f8')
            info['matrix_shape']=list(matrix.shape)
            info['matrix_float64_sha256']=hashlib.sha256(matrix.tobytes()).hexdigest()
    return result


def archive(output=HERE/'results'):
    original=output/'records.jsonl.gz';target=output/'records.compact.jsonl.gz'
    if target.exists():raise RuntimeError('Canonical evidence already exists; no rewrite')
    count=0
    with gzip.open(original,'rt') as source, target.open('wb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as dest:
        for line in source:
            dest.write((json.dumps(compact(json.loads(line)),allow_nan=False)+'\n').encode());count+=1
    save(output/'projection.json',dict(records=count,
        original_full_archive_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        original_bytes=original.stat().st_size,canonical_bytes=target.stat().st_size,
        canonical_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        omitted='Only deterministic dense observed-information matrices; shapes/eigenvalues/conditions/matrix SHA256 retained. All fits/parameters/LR/p/decisions/failures/gradient vectors/RNG and data hashes preserved.',
        original_location='Full original archive retained in execution workspace; not modified. Matrices reproducible from saved parameters, pinned source/runtime and verified data; no optimization required.'))

if __name__=='__main__':archive()
