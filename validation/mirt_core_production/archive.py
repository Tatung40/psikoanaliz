"""Read canonical NRM evidence only; NEVER fits or auto-recreates missing files."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def archive(root):
    root=Path(root)
    report=json.loads((root/"report.json").read_text())
    settings=json.loads((root/"settings.json").read_text())
    assert report["protocol"]["commit"]=="af024e73e4be99d0db1a2a0df400039f98ab6095"
    assert settings["criteria_sha256"]=="5b666c899ee9cc53e3683b99fe6d192816fdcc0731ca0a32b61607f612910d44"
    assert report["calibration_attempts"]["Python"]<=1 and report["calibration_attempts"]["R"]<=1
    files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    assert files["data.csv"]==settings["data_sha256"],"Canonical response matrix changed"
    project=Path(__file__).resolve().parents[2]
    sources=json.loads((root/"sources.json").read_text())
    for path,expected in sources.items():
        assert hashlib.sha256((project/path).read_bytes()).hexdigest()==expected,path
    payload={"report":report,"settings":settings,"file_sha256":files,"new_fits":0,
             "archive_note":"Original actual run evidence and raw native coefficients; no calibration or substitute estimates."}
    for filename in ("data.csv","cold_starts.json","sources.json","trace.csv","python_fit.json","python_metadata.json",
                     "fit/parameters.csv","fit/raw_parameters.csv","fit/metadata.json","fit/eap.csv",
                     "fit/probabilities.csv","fit/information.csv"):
        path=root/filename
        if path.is_file():payload[filename]=path.read_text()
    params=root/"python_parameters.npz"
    if params.is_file():
        with np.load(params,allow_pickle=False) as p:payload["python_parameters"]={k:p[k].tolist() for k in p.files}
    return payload

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);args=parser.parse_args()
    print("NRM_CANONICAL_ARCHIVE="+json.dumps(archive(args.root),ensure_ascii=False,allow_nan=False))
