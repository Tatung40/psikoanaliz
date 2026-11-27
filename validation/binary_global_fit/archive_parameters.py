
"""Archive saved binary estimates, without any estimation or numerical recomputation."""
import argparse,json,hashlib,base64
from pathlib import Path
import numpy as np
def main():
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--out",required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for model in ("3PL","4PL"):
        paths=sorted(Path(a.input).rglob(model+"_*.json"))
        rows=[json.loads(p.read_text()) for p in paths]
        assert len(rows)==7800 and all("eta" in r for r in rows)
        eta=np.asarray([r["eta"] for r in rows],dtype="<f8")
        for path,row,e in zip(paths,rows,eta):
            saved=np.load(path.with_suffix(".npz"),allow_pickle=False)
            assert np.array_equal(saved["eta"],e)
        target=out/("COEFFICIENTS_"+model+".npz")
        np.savez_compressed(target,eta=eta,r=np.asarray([r["r"] for r in rows],dtype="<i4"),
                            kind=np.asarray([("null","nonmonotone","local_dependence","second_factor").index(r["kind"]) for r in rows],dtype="u1"))
        raw=target.read_bytes()
        print("PARAMETER_ARCHIVE "+json.dumps(dict(model=model,shape=list(eta.shape),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
              git_blob=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest())),flush=True)
if __name__=="__main__":main()
