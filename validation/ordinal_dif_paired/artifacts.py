"""Lossless artifact cleanup: every removed byte stream has a hashed recipe."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
MANIFEST=Path(__file__).with_name('artifact_audit.json')
BASE='5da63fa9a8885c6b1b572727dfd3487e152ceb98'
MAIN='eb8c798'


def sha(data):return hashlib.sha256(data).hexdigest()


def reconstruct(entry, root=ROOT):
    recipe=entry['reconstruct']
    if recipe['type']=='empty':return b''
    if recipe['type']=='copy':return (root/recipe['path']).read_bytes()
    value=json.loads((root/recipe['path']).read_text())
    for key in recipe['keys']:value=value[key]
    return json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8')


def audit_and_prune():
    paths=subprocess.check_output(['git','diff','--name-only',MAIN+'...'+BASE],cwd=ROOT,text=True).splitlines()
    entries=[]
    for name in paths:
        raw=subprocess.check_output(['git','show',BASE+':'+name],cwd=ROOT)
        entry=dict(path=name,bytes=len(raw),sha256=sha(raw),action='retained')
        p=ROOT/name;recipe=None
        if name.endswith(('.stdout','.stderr')) and not raw:recipe=dict(type='empty')
        elif '/grm_' in name and name.endswith('.json'):
            seed=p.stem[4:];record=p.with_name('rep_'+seed+'.json')
            recipe=dict(type='json_subtree',path=str(record.relative_to(ROOT)),keys=['methods','GRM_LR'])
        elif '/evidence_initial_api/' in name and (p.name in ('mechanism.json','precision_plan.json') or p.suffix=='.csv'):
            recipe=dict(type='copy',path=str(p.parent.with_name('evidence')/p.name))
            recipe['path']=str(Path(recipe['path']).relative_to(ROOT))
        if recipe is not None:
            entry['reconstruct']=recipe
            if p.read_bytes()!=raw or reconstruct(entry)!=raw:
                raise ValueError('Cannot remove nonidentical evidence: '+name)
            entry['action']='removed_losslessly';p.unlink()
        entries.append(entry)
    duplicate_raw=[]
    for i in range(2000100,2000110):
        a=ROOT/f'validation/ordinal_dif_paired/evidence/data_{i}.csv'
        b=ROOT/f'validation/ordinal_dif_paired/evidence_initial_api/data_{i}.csv'
        entry=next(e for e in entries if e['path']==str(b.relative_to(ROOT)))
        if sha(a.read_bytes())!=entry['sha256']:raise ValueError('Original paired CSV mismatch')
        duplicate_raw.append(dict(canonical=str(a.relative_to(ROOT)),reconstructable_alias=str(b.relative_to(ROOT)),sha256=sha(a.read_bytes())))
    result=dict(audited_commit=BASE,files=len(entries),bytes=sum(e['bytes'] for e in entries),
                removed=sum(e['action']=='removed_losslessly' for e in entries),entries=entries,
                deduplicated_csv=duplicate_raw,
                rationale='One canonical copy per raw matrix; restore byte-identical historical aliases on demand. No oversized temporary data. All removed evidence is byte-reconstructable.')
    MANIFEST.write_text(json.dumps(result,indent=2));return result


def verify(restore=False):
    manifest=json.loads(MANIFEST.read_text());count=0
    for entry in manifest['entries']:
        if entry['action']!='removed_losslessly':continue
        raw=reconstruct(entry)
        if len(raw)!=entry['bytes'] or sha(raw)!=entry['sha256']:raise ValueError('Artifact recipe mismatch')
        if restore:
            path=ROOT/entry['path']
            if path.exists() and path.read_bytes()!=raw:raise ValueError('Refusing to overwrite altered evidence')
            path.write_bytes(raw)
        count+=1
    return count


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prune',action='store_true');parser.add_argument('--restore',action='store_true')
    args=parser.parse_args()
    print(audit_and_prune() ['removed'] if args.prune else verify(args.restore))
