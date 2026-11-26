"""Read-only committed selected-Q evidence audit. NO data generation, fits or randomization."""
import gzip,hashlib,json,math,statistics
from pathlib import Path
from statsmodels.stats.proportion import proportion_confint
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"validation/evidence/p3_4pl_closure"
KINDS=("null","nonmonotone","local_dependence","two_stage")
def bound(k,n,a,side):
    return float(proportion_confint(k,n,alpha=2*a,method="beta")[1 if side=="upper" else 0])

def audit():
    proof=json.loads((BASE/"AUDIT.json").read_text());raw=(BASE/"CONFIRMATION.jsonl.gz").read_bytes()
    assert hashlib.sha256(raw).hexdigest()==proof["archive_sha256"]
    rows=[json.loads(line) for line in gzip.decompress(raw).splitlines()]
    cfg=json.loads((ROOT/"validation/p3_4pl_closure/PROTOCOL.json").read_text())["confirmation"]
    assert len(rows)==7800 and len({(r["kind"],r["r"]) for r in rows})==7800
    by={k:sorted((r for r in rows if r["kind"]==k),key=lambda r:r["r"]) for k in KINDS}
    for kind,rr in by.items():
        count=6000 if kind=="null" else 600;root=cfg["null_root"] if kind=="null" else cfg["power_root"]
        assert [r["r"] for r in rr]==list(range(count))
        for r in rr:
            assert r["seed"]==[root,4,r["r"],KINDS.index(kind)]
            assert r["source"]==proof["source"] and r["run"]==proof["run"] and r["mirt"]=="1.2.0"
            assert len(r["data_sha256"])==64 and r["model"]=="4PL"
            if "p_grid" in r:
                assert len(r["p_grid"])==len(r["Q"])==len(r["sx"])==len(r["sx_native"])==len(r["native_valid"])==12
            for g in r.get("p_grid",[]):assert g is None or isinstance(g,int) and 1<=g<=4096
    def values(r):
        return [(g/4096 if g is not None and v else None) for g,v in zip(r["p_grid"],r["native_valid"])] if "p_grid" in r else [None]*12
    event=[[p is None or p<.05 for p in values(r)] for r in by["null"]]
    rates=[sum(row)/12 for row in event]
    counts=[sum(row[j] for row in event) for j in range(12)]
    mcse=statistics.stdev(rates)/math.sqrt(6000);size=statistics.mean(rates)
    ups=[bound(k,6000,.05/24,"upper") for k in counts]
    identity=max(max(abs(x-y) for x,y in zip(r["sx"],r["sx_native"])) for r in rows if "sx" in r)
    max_kkt=max(r["max_KKT"] for r in rows if "max_KKT" in r)
    refusal=sum(any(p is None for p in values(r)) for r in by["null"])/6000
    assert counts==proof["item_counts"]
    for actual,expected in ((size,proof["TypeI"]),(mcse,proof["MCSE"]),(max(ups),proof["worst_upper"]),(identity,proof["identity_max"])):
        assert math.isclose(actual,expected,rel_tol=0,abs_tol=1e-14)
    powers={}
    for kind in KINDS[1:]:
        vals=[values(r)[0] for r in by[kind]];k=sum(v is not None and v<.05 for v in vals)
        low=bound(k,600,.025/3,"lower");invalid=sum(v is None for v in vals)/600
        assert k==proof["power"][kind]["rejections"]
        assert math.isclose(low,proof["power"][kind]["lower"],rel_tol=0,abs_tol=1e-14)
        powers[kind]=dict(power=k/600,lower=low,refusal=invalid)
    gates=dict(size=max(ups)<=.07,MCSE=mcse<=.004,pooled_lower=size-1.959963984540054*mcse>=.03,
        refusal=refusal<=.01,identity=identity<=1e-8,complete=True,selected_before_confirmation=True)
    claims=[k for k,v in powers.items() if all(gates.values()) and v["lower"]>.10 and v["refusal"]<=.01]
    assert gates==proof["gates"] and claims==proof["certified_claims"] and max_kkt<=1e-7
    cert=json.loads((BASE/"CERTIFICATION.json").read_text())
    assert cert["certified_claims"]==claims and cert["gates"]==gates and cert["status"]=="SCOPED CERTIFIED"
    assert cert["checkout_commit"]==proof["source"] and cert["confirmation_run"]==int(proof["run"])
    preserved=json.loads((ROOT/"validation/p3_4pl_closure/PRESERVED.json").read_text())
    for path,expected in preserved.items():
        b=(ROOT/path).read_bytes()
        assert hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()==expected,path
    return dict(records=len(rows),no_new_fits=True,TypeI=size,MCSE=mcse,worst_upper=max(ups),
        power=powers,refusal=refusal,max_KKT=max_kkt,identity_max=identity,certified_claims=claims,preserved_certificates=21)
if __name__=="__main__":print(json.dumps(audit()))
