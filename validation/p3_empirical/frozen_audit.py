"""Read-only audit of committed P3 confirmation decisions. No fitting or data generation."""
import gzip,hashlib,json,math,statistics
from pathlib import Path
from scipy.stats import beta,norm


def audit(root=None):
    root=Path(root) if root else Path(__file__).resolve().parents[2]
    folder=root/"validation/evidence/p3_empirical"
    archived=json.loads((folder/"AUDIT.json").read_text())
    raw=(folder/"CONFIRMATION.jsonl.gz").read_bytes()
    assert hashlib.sha256(raw).hexdigest()==archived["raw_archive_sha256"]
    assert hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()==archived["raw_archive_git_blob"]
    rows=[json.loads(l) for l in gzip.decompress(raw).splitlines()]
    assert len(rows)==6600
    null=[r for r in rows if not r["alt"]];alt=[r for r in rows if r["alt"]]
    for group,rootseed,n in ((null,202611262001,6000),(alt,202611262002,600)):
        assert len(group)==n and {r["r"] for r in group}==set(range(n))
        for r in group:
            assert r["seed"]==[rootseed,3,r["r"],0] and r["root"]==rootseed
            assert r["model"]=="3PL" and r["source"]=="e978d102e86157c044f2bc50ae2fe6b28f6d0c1e"
            assert r["converged"] and not r.get("error")
            assert r["identity"]<=1e-8 and len(r["data_sha256"])==64
            assert all(math.isfinite(v) and v>0 for v in r["df"])
            assert all(math.isfinite(v) for v in r["sx_native"])
            assert all(math.isfinite(p) and 1/4096<=p<=1 and round(p*4096)==p*4096 for p in r["p_constrained"])
    counts=[sum(r["p_constrained"][j]<.05 for r in null) for j in range(12)]
    cp=[float(beta.ppf(1-.05/24,k+1,6000-k)) for k in counts]
    clusters=[sum(p<.05 for p in r["p_constrained"])/12 for r in null]
    rate=statistics.mean(clusters);mcse=statistics.stdev(clusters)/math.sqrt(6000)
    power=sum(r["p_constrained"][0]<.05 for r in alt)
    lower=float(beta.ppf(.025,power,600-power+1))
    assert counts==archived["rejections_per_item"] and power==archived["power_rejections"]
    assert abs(rate-archived["typeI"])<1e-12 and abs(mcse-archived["MCSE"])<1e-12
    assert abs(max(cp)-archived["worst_upper"])<1e-12 and abs(lower-archived["power_lower"])<1e-12
    assert max(cp)<=.07 and mcse<=.004 and rate-float(norm.ppf(.975))*mcse>=.03 and lower>.10
    proof=json.loads((folder/"CERTIFICATION.json").read_text())
    assert proof["status"]=="SCOPED CERTIFIED" and proof["model"]=="3PL"
    assert proof["confirmation_run"]==45808993982 and proof["confirmation_head"]=="88afec3341859b71ca970fb064b2d2159f382fe8"
    for path,expected in proof["scientific_source_blobs"].items():
        source=(root/path).read_bytes()
        assert hashlib.sha1(b"blob "+str(len(source)).encode()+b"\0"+source).hexdigest()==expected
    return {"records":len(rows),"all_gates":"PASS","typeI":rate,"MCSE":mcse,"worst_upper":max(cp),"power":power/600,"power_lower":lower,"no_fits":True}


if __name__=="__main__":print(json.dumps(audit()))
