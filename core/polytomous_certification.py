"""Evidence-bound P4-C1 labels; no extrapolation or invented uncertainty."""
import hashlib,json
from pathlib import Path
from importlib.metadata import version,PackageNotFoundError
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/"validation/evidence/p4_confirm/REPORT.json"
PROTOCOL=ROOT/"validation/p4_confirm/PROTOCOL.json"
RUN=ROOT/"validation/p4_confirm/RUN.json"
BLOCKS={"GRM":{"discrimination","thresholds"},"PCM":{"steps"},"GPCM":{"discrimination","steps"},"NRM":{"slopes","intercepts"}}
SETTINGS=dict(max_iter=500,tol=1e-4,n_quadpts=21)

def _evidence_fingerprint():
    h=hashlib.sha256()
    for path in (Path(__file__),ROOT/"core/polytomous_inference.py",REPORT,PROTOCOL,RUN):
        try:h.update(path.read_bytes())
        except OSError:h.update(b"MISSING")
    return h.hexdigest()

def _confirmed(model,block):
    try:
        protocol=json.loads(PROTOCOL.read_text());run=json.loads(RUN.read_text())
        c=json.loads(REPORT.read_text())[model+" "+block]
        if version("mirt")!="1.2.0" or not protocol["registered_before_confirmation"] or run["run_id"]!=45853181770:return False
        # Exact registered P4-C1 gates; not tuned to any new result.
        n=c["R"];p=c["coverage"];se=c["cluster_mcse"];refusal=c["refusal"]
        return (n==protocol["R"]["value"]==400 and c["CERTIFIED"] is True and
                all(v is True for v in c["gates"].values()) and .925<=p<=.975 and
                p-2.690*se>=.925 and p+2.690*se<=.975 and 0<=se<=.004 and 0<=refusal<=.01)
    except (OSError,ValueError,KeyError,TypeError,PackageNotFoundError):return False

def ci_status(model,block,n_persons,estimation="ML",solver_settings=None,se_method=None,n_items=None,n_categories=None):
    ok=(n_items==10 and n_categories==4 and model in BLOCKS and block in BLOCKS[model] and n_persons==2000 and
        estimation=="ML" and dict(solver_settings or {})==SETTINGS and (model!="NRM" or se_method=="oakes") and _confirmed(model,block))
    return "SCOPED CERTIFIED (P4-C1: N=2000, kayıtlı üreteç, varsayılan ML ayarları)" if ok else "NOT CERTIFIED"
