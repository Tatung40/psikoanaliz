"""Independent arithmetic audit + compact immutable archive; no item-model fit."""
import argparse,base64,gzip,hashlib,json,math,statistics
from pathlib import Path
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from validation.p3_4pl_closure.dgp import KINDS,data
ROOT=Path(__file__).resolve().parents[2]
def upper(k,n,a):return float(proportion_confint(k,n,alpha=2*a,method="beta")[1])
def lower(k,n,a):return float(proportion_confint(k,n,alpha=2*a,method="beta")[0])
def evaluate(rows):
    cfg=json.loads((ROOT/"validation/p3_4pl_closure/PROTOCOL.json").read_text())["confirmation"]
    by={k:sorted([r for r in rows if r["kind"]==k],key=lambda r:r["r"]) for k in KINDS}
    assert len(rows)==7800 and len({(r["kind"],r["r"]) for r in rows})==7800
    sources={r["source"] for r in rows};runs={r["run"] for r in rows};assert len(sources)==len(runs)==1
    for kind,rr in by.items():
        count=6000 if kind=="null" else 600;root=cfg["null_root"] if kind=="null" else cfg["power_root"]
        assert [r["r"] for r in rr]==list(range(count))
        for r in rr:
            assert r["seed"]==[root,4,r["r"],KINDS.index(kind)] and r["mirt"]=="1.2.0"
        # Regenerate a predeclared audit subset, never recalibrate.
        for r in rr[:3]:
            x,_=data(root,r["r"],kind)
            assert hashlib.sha256(x.tobytes()).hexdigest()==r["data_sha256"]
    def probabilities(rr):
        p=[]
        for r in rr:
            if r.get("result") and not r.get("error"):
                pv=[v["p_constrained_standardized"] for v in r["result"]["items"]]
                p.append([v if v is not None and native is not None else None for v,native in zip(pv,r["p_chi"])])
            else:p.append([None]*12)
        return p
    p=probabilities(by["null"]);invalid=lambda v:v is None or not math.isfinite(v)
    event=[[invalid(v) or v<.05 for v in row] for row in p]
    counts=[sum(row[j] for row in event) for j in range(12)];rates=[sum(row)/12 for row in event]
    mcse=statistics.stdev(rates)/math.sqrt(6000);rate=statistics.mean(rates)
    ups=[upper(k,6000,.05/24) for k in counts]
    powers={}
    for kind in KINDS[1:]:
        pp=[r[0] for r in probabilities(by[kind])];k=sum(not invalid(v) and v<.05 for v in pp)
        powers[kind]=dict(R=600,rejections=k,power=k/600,lower=lower(k,600,.025/3),upper=upper(k,600,.025/3),
            refusal=sum(invalid(v) for v in pp)/600)
    identities=[max(abs(x["sx"]-y) for x,y in zip(r["result"]["items"],r["sx_native"]))
                for r in rows if r.get("result") and not r.get("error")]
    refusal=sum(any(invalid(v) for v in row) for row in p)/6000
    gates=dict(size=max(ups)<=.07,MCSE=mcse<=.004,pooled_lower=rate-1.959963984540054*mcse>=.03,
        refusal=refusal<=.01,identity=max(identities)<=1e-8,complete=True,selected_before_confirmation=True)
    claims=[kind for kind,v in powers.items() if all(gates.values()) and v["lower"]>.10 and v["refusal"]<=.01]
    return dict(independent_arithmetic="PASS",null_R=6000,power_R_each=600,TypeI=rate,MCSE=mcse,item_counts=counts,
        upper=ups,worst_upper=max(ups),power=powers,refusal=refusal,identity_max=max(identities),
        all_native_converged=sum(bool(r.get("converged")) for r in rows),records=len(rows),source=next(iter(sources)),run=next(iter(runs)),
        gates=gates,certified_claims=claims,no_new_fits=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",required=True);ap.add_argument("--out",required=True);z=ap.parse_args()
    rows=[]
    for f in sorted(Path(z.input).rglob("4PL_*.json")):
        rows.append(json.loads(f.read_text()))
    report=evaluate(rows);dest=Path(z.out);dest.mkdir(parents=True,exist_ok=True)
    for r in rows:r.pop("eta",None) # complete native coefs + pattern counts remain the original .npz artifacts
    raw="".join(json.dumps(r,separators=(",",":"))+"\n" for r in rows).encode()
    archive=gzip.compress(raw,mtime=0);(dest/"CONFIRMATION.jsonl.gz").write_bytes(archive)
    report["archive_sha256"]=hashlib.sha256(archive).hexdigest()
    expected=hashlib.sha1(b"blob "+str(len(archive)).encode()+b"\0"+archive).hexdigest();report["archive_git_blob"]=expected
    (dest/"AUDIT.json").write_text(json.dumps(report,indent=1))
    print("INDEPENDENT_AUDIT_JSON "+json.dumps(report),flush=True)
    b64=base64.b64encode(archive).decode()
    print("ARCHIVE_META "+json.dumps(dict(bytes=len(archive),chunks=math.ceil(len(b64)/100000),git_blob=expected)),flush=True)
    for i in range(0,len(b64),100000):print("ARCHIVE_CHUNK "+str(i//100000)+" "+b64[i:i+100000],flush=True)
if __name__=="__main__":main()
