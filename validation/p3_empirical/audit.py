"""Independent arithmetic/provenance audit of fixed confirmation records; never fits."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",required=True);ap.add_argument("--report",required=True);ap.add_argument("--out",required=True)
    z=ap.parse_args();rows=[json.loads(f.read_text()) for f in Path(z.input).rglob("3PL_*.json")]
    null=[r for r in rows if not r["alt"]];alt=[r for r in rows if r["alt"]]
    assert len(null)==6000 and len(alt)==600
    assert {r["r"] for r in null}==set(range(6000)) and {r["r"] for r in alt}==set(range(600))
    for group,root in ((null,202611262001),(alt,202611262002)):
        for r in group:
            assert r["seed"]==[root,3,r["r"],0] and r["root"]==root
            assert r["source"]=="e978d102e86157c044f2bc50ae2fe6b28f6d0c1e"
            assert len(r["data_sha256"])==64
    rej=[];invalid=0;identity=0.
    for r in null:
        values=[]
        for j in range(12):
            p=r.get("result",{}).get("items",[{}]*12)[j].get("p_constrained")
            valid=not r.get("error") and p is not None and math.isfinite(p) and math.isfinite(r["p_chi"][j])
            invalid+=not valid;values.append((not valid) or p<.05)
            if valid:assert 1/4096<=p<=1 and round(p*4096)==p*4096
        rej.append(values)
        if not r.get("error"):
            assert r["native_identity_available"]
            identity=max(identity,r["identity"])
    native_converged=sum(bool(r.get("converged")) for r in rows)
    for r in rows:
        if r.get("error"):continue
        assert r["native_identity_available"] and len(r["sx_native"])==12
        d=max(abs(a-item["sx"]) for a,item in zip(r["sx_native"],r["result"]["items"]))
        assert d<=1e-8
        identity=max(identity,d)
    counts=[sum(row[j] for row in rej) for j in range(12)]
    cp=[float(proportion_confint(k,6000,alpha=2*(.05/24),method="beta")[1]) for k in counts]
    cluster=[sum(r)/12 for r in rej];rate=statistics.mean(cluster);mcse=statistics.stdev(cluster)/math.sqrt(6000)
    power=0
    for r in alt:
        p=r.get("result",{}).get("items",[{}])[0].get("p_constrained")
        power+=not r.get("error") and p is not None and math.isfinite(p) and p<.05
    lower=float(proportion_confint(power,600,alpha=.05,method="beta")[0])
    source=json.loads(Path(z.report).read_text())["models"]["3PL"]["p_constrained"]
    assert abs(source["typeI"]-rate)<1e-12 and abs(source["MCSE"]-mcse)<1e-12
    assert max(abs(a-b) for a,b in zip(cp,source["upper"]))<1e-12
    assert abs(source["power_lower"]-lower)<1e-12
    result={"independent_arithmetic":"PASS","null_R":6000,"power_R":600,"rejections_per_item":counts,
        "power_rejections":power,"typeI":rate,"MCSE":mcse,"worst_upper":max(cp),"power":power/600,"power_lower":lower,
        "native_converged":native_converged,"identity_max":identity,"refusal":invalid/(6000*12),"all_seeds_and_sources_verified":True,
        "decision":bool(max(cp)<=.07 and mcse<=.004 and rate-float(norm.ppf(.975))*mcse>=.03 and invalid/(6000*12)<=.01 and identity<=1e-8 and lower>.10)}
    import gzip,base64
    compact=[]
    for r in sorted(rows,key=lambda r:(r["root"],r["r"])):
        items=r.get("result",{}).get("items",[{}]*12)
        record={k:r.get(k) for k in ("model","root","r","alt","seed","source","run","data_sha256","converged","iterations","error","identity","seconds","sx_native","df")}
        record["p_constrained"]=[v.get("p_constrained") for v in items]
        record["reference_rank"]=[v.get("rank") for v in items]
        record["QP_diagnostics"]=r.get("result",{}).get("diagnostics")
        compact.append(record)
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list):return [clean(x) for x in v]
        if isinstance(v,float) and not math.isfinite(v):return None
        return v
    raw=("\n".join(json.dumps(clean(r),separators=(",",":"),allow_nan=False) for r in compact)+"\n").encode()
    packed=gzip.compress(raw,mtime=0)
    Path(z.out).with_name("CONFIRMATION.jsonl.gz").write_bytes(packed)
    result.update(raw_archive_sha256=hashlib.sha256(packed).hexdigest(),raw_archive_rows=len(compact),
        raw_archive_git_blob=hashlib.sha1(b"blob "+str(len(packed)).encode()+b"\0"+packed).hexdigest())
    Path(z.out).write_text(json.dumps(result,indent=1));print("INDEPENDENT_AUDIT_JSON "+json.dumps(result))
    encoded=base64.b64encode(packed).decode()
    for i in range(0,len(encoded),50000):print("ARCHIVE_CHUNK "+str(i//50000)+" "+encoded[i:i+50000],flush=True)



if __name__=="__main__":main()
