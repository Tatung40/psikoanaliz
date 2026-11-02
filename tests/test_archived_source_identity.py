"""Protect all archived bytes; no generic historical/hash fallback."""
from pathlib import Path
import hashlib
from validation.archived_source_identity import LEGACY_SIZE,LEGACY_SHA,source_matches


def test_exact_legacy_prefix_and_only_reviewed_extension(tmp_path):
    data=Path('core/irt_dif.py').read_bytes()
    assert hashlib.sha256(data[:LEGACY_SIZE]).hexdigest()==LEGACY_SHA
    assert source_matches('core/irt_dif.py',LEGACY_SHA)
    core=tmp_path/'core';core.mkdir();path=core/'irt_dif.py'
    for bad in [data.replace(b'LR',b'XX',1),data+b'\nSCORE_TOLERANCE = 1\n',
                data+b'\ndef dif_analysis(): pass\n',data.replace(b'kinds=None',b'kinds=print(1)'),
                data+b'\ndef extra(): pass\n']:
        path.write_bytes(bad)
        assert not source_matches(path,LEGACY_SHA)
    path.write_bytes(data[:LEGACY_SIZE])
    assert source_matches(path,LEGACY_SHA)
    # An unrelated file or a different expected identity never receives this exception.
    other=tmp_path/'unrelated.py';other.write_bytes(data)
    assert not source_matches(other,LEGACY_SHA)
    assert not source_matches('core/irt_dif.py','0'*64)


def test_frozen_auditor_view_never_changes_data_or_other_source_hashes():
    from types import SimpleNamespace
    from validation.archived_source_identity import historical_digest
    module=SimpleNamespace(digest=lambda b:hashlib.sha256(b).hexdigest())
    original=module.digest;source=Path('core/irt_dif.py').read_bytes()
    payload=b'unchanged raw scientific record'
    with historical_digest(module):
        assert module.digest(source)==LEGACY_SHA
        assert module.digest(payload)==hashlib.sha256(payload).hexdigest()
        changed=source.replace(b'LR',b'XX',1)
        assert module.digest(changed)!=LEGACY_SHA
    assert module.digest is original
    assert module.digest(source)==hashlib.sha256(source).hexdigest()
