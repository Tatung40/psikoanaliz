
"""Independent frozen-ledger statistics audit: SciPy CP, no fits/R/references rerun."""
import json,gzip,hashlib,math
from pathlib import Path
import numpy as np
from scipy.stats import beta
ROOT=Path(__file__).resolve().parents[2]
E=ROOT/"validation/evidence/binary_global_fit"
def interval(k,n,a):
    return (float(beta.ppf(a,k,n-k+1)) if k else 0.,
            float(beta.ppf(1-a,k+1,n-k)) if k<n else 1.)
def main():
    proof=json.loads((E/"CERTIFICATION.json").read_text())
    raw=(E/"CONFIRMATION.jsonl.gz").read_bytes()
    assert hashlib.sha256(raw).hexdigest()==proof["ledger_sha256"]
    records=[json.loads(line) for line in gzip.decompress(raw).decode().splitlines()]
    assert len(records)==15600
    assert len({(r["model"],r["kind"],r["r"]) for r in records})==15600
    assert len({tuple(r["seed"]) for r in records})==15600
    assert len({r["data_sha256"] for r in records})==15600
    assert {r["source"] for r in records}=={proof["canonical_source"]}
    for p,sha in proof["scientific_source_blobs"].items():
        b=(ROOT/p).read_bytes();assert hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()==sha,p
    audit={}
    for model in ("3PL","4PL"):
        cell=proof["models"][model];method=cell["method"];rr=[r for r in records if r["model"]==model]
        null=[r for r in rr if r["kind"]=="null"];assert {r["r"] for r in null}==set(range(6000))
        events=[];bad=[]
        for r in null:
            p=r.get("result",{}).get(method)
            fail=bool(r.get("error")) or p is None or not np.isfinite(p)
            bad.append(fail);events.append(fail or p<.05)
            if not fail:
                assert r["converged"] and r["result"]["max_KKT"]<=1e-7
                assert r["identity"]<=1e-7+1e-6*abs(r["result"]["M2"])
        k=sum(events);n=len(null);rate=k/n;lo,hi=interval(k,n,.025);se=math.sqrt(rate*(1-rate)/n)
        for expected,actual in ((cell["TypeI"],rate),(cell["lower"],lo),(cell["upper"],hi),(cell["MCSE"],se),(cell["refusal"],sum(bad)/n)):
            assert np.isclose(expected,actual,rtol=0,atol=1e-12),(model,expected,actual)
        assert cell["gates"]==dict(size=hi<=.07,nontrivial=lo>=.03,MCSE=se<=.004,refusal=sum(bad)/n<=.01,
                                   convergence=sum(bool(r["converged"]) for r in null)/n>=.99,complete=True)
        claims=[]
        for kind in ("nonmonotone","local_dependence","second_factor"):
            rows=[r for r in rr if r["kind"]==kind];assert {r["r"] for r in rows}==set(range(600))
            pp=[r.get("result",{}).get(method) for r in rows]
            fails=[bool(r.get("error")) or p is None or not np.isfinite(p) for r,p in zip(rows,pp)]
            hits=sum(not f and p<.05 for f,p in zip(fails,pp));lower,upper=interval(hits,600,.025/6)
            reported=cell["power"][kind];assert reported["k"]==hits and reported["R"]==600
            assert np.isclose(reported["lower"],lower,rtol=0,atol=1e-12)
            assert np.isclose(reported["upper"],upper,rtol=0,atol=1e-12)
            if all(cell["gates"].values()) and lower>.10 and sum(fails)/600<=.01:claims.append(kind)
        assert claims==cell["certified_claims"]
        assert (cell["status"]=="SCOPED CERTIFIED")==bool(claims)
        audit[model]=dict(null_events=k,R=n,TypeI=rate,CP=[lo,hi],MCSE=se,certified_claims=claims)
    print("INDEPENDENT_AUDIT_JSON "+json.dumps(audit))
if __name__=="__main__":main()
