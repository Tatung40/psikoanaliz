
"""Lossless repeat-level decision ledger, generated from frozen fits; ZERO refits."""
import json,gzip,hashlib,base64,argparse,os,sys,platform
from pathlib import Path
from importlib.metadata import version
import numpy as np
from validation.binary_global_fit.statistics import summary
KINDS=("null","nonmonotone","local_dependence","second_factor")
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",required=True);ap.add_argument("--out",required=True);z=ap.parse_args()
    out=Path(z.out);out.mkdir(parents=True,exist_ok=True)
    paths=sorted(Path(z.input).rglob("*.json"))
    paths=[p for p in paths if p.name.startswith(("3PL_","4PL_"))]
    records=[json.loads(p.read_text()) for p in paths]
    assert len({(r["model"],r["kind"],r["r"]) for r in records})==len(records)
    thin=[]
    for r,path in zip(records,paths):
        result=r.get("result",{})
        # All candidate p values retained; exact estimates/counts remain in original canonical checkpoint artifacts.
        keep={k:r.get(k) for k in ("model","kind","r","root","seed","source","run","data_sha256","protocol_commit","mirt","converged","iterations","error","seconds","identity")}
        keep["result"]={k:result.get(k) for k in ("M2","df","p_chi","p_linear","p_nonlinear","p_QLI","QLI","seconds","max_KKT","active_fraction","QLI_rank","nullspace_residual","mean_norm")}
        if "rank" in result:
            keep["result"]["rank"]={k:v for k,v in result["rank"].items() if k!="jacobian_singular_values"}
        if "eta" in r:
            a=np.asarray(r["eta"],dtype="<f8")
            keep["eta_sha256"]=hashlib.sha256(a.tobytes()).hexdigest()
            checkpoint=path.with_suffix(".npz")
            assert checkpoint.exists(),"Missing frozen parameter/count checkpoint; never refit it"
            saved=np.load(checkpoint,allow_pickle=False)
            assert np.array_equal(saved["eta"],a)
            assert saved["counts"].shape==(4096,) and int(saved["counts"].sum())==3000
            keep["checkpoint_sha256"]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            keep["counts_sha256"]=hashlib.sha256(np.asarray(saved["counts"],dtype="<u2").tobytes()).hexdigest()
        thin.append(keep)
    body=("\n".join(json.dumps(r,sort_keys=True,separators=(",",":")) for r in thin)+"\n").encode()
    target=out/"CONFIRMATION.jsonl.gz";target.write_bytes(gzip.compress(body,compresslevel=9,mtime=0))
    select=json.loads(Path("validation/binary_global_fit/SELECTION.json").read_text())
    report={m:summary(thin,m,select["methods"][m],"confirm") for m in ("3PL","4PL")}
    (out/"REPORT.json").write_text(json.dumps(report,indent=2))
    scientific=("validation/binary_global_fit/kernel.py","validation/binary_global_fit/analysis.py","validation/binary_global_fit/dgp.py","validation/binary_global_fit/runner.py","validation/binary_global_fit/statistics.py","validation/binary_certification/common.py","validation/binary_global_fit/PROTOCOL.json","validation/binary_global_fit/AMENDMENT_A1.json","validation/binary_global_fit/AMENDMENT_A2.json","validation/binary_global_fit/SELECTION.json","validation/binary_global_fit/dgp.py","validation/binary_global_fit/runner.py","validation/binary_global_fit/statistics.py","validation/p3_empirical/kernel.py","validation/p3_empirical/constrained.py","validation/p3_sx2_analytic/kernel.py")
    blobs={}
    for p in scientific:
        b=Path(p).read_bytes();blobs[p]=hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
    cert=dict(version="binary-global-fit-O1",canonical_source="f3d8d24d76c45e48b5836ba057a2c1d4aab4e441",canonical_pr_head="1d166c5310c34136df51f8e58fb09630206512e9",canonical_main="69be9aa4f11590662a49f122198db75dbeced7dc",canonical_run=45856527113,
              protocol_commit="521be0af635198e7bc0ab326b47e0ab7a066ca86",models={},scientific_source_blobs=blobs,
              ledger_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),software={p:version(p) for p in ("mirt","numpy","scipy","pandas","statsmodels")},
              canonical_software={"Python":"3.11.18","mirt":"1.2.0","numpy":"2.4.6","scipy":"1.17.1","pandas":"3.1.1","statsmodels":"0.15.0"},
              canonical_environment_provenance="Original shard5 job140037432792 install log, same Python3.11/setup/cache and requirements across24 shards",
              Python=sys.version,platform=platform.platform(),archive="All raw fitted coefficients/counts/complete candidate diagnostics preserved in the24 canonical global-confirm-shard artifacts; ledger retains every attempted decision/error/hash and coefficient hash. No archived commit may refit missing evidence.",
              scope="Reference true parameters only; N3000/J12 fixed N(0,1), native cold MAP default priors/GH61/max500/tol1e-4. Not a user-DGP guarantee, no FWER/FDR, no RMSEA/CFI/TLI interval certificate.")
    for m,r in report.items():
        cert["models"][m]={**r,"status":"SCOPED CERTIFIED" if r["certified_claims"] else "NOT CERTIFIED",
                           "method_label":"Native M2 / joint constrained "+("linear" if m=="3PL" else "nonlinear")+" Gaussian MAP reference"}
    (out/"CERTIFICATION.json").write_text(json.dumps(cert,indent=2))
    print("ARCHIVE_REPORT_JSON "+json.dumps(report),flush=True)
    print("ARCHIVE_CERTIFICATION_JSON "+json.dumps(cert),flush=True)
    raw=target.read_bytes();gitsha=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
    print("ARCHIVE_LEDGER_INFO "+json.dumps(dict(size=len(raw),git_blob=gitsha,sha256=cert["ledger_sha256"])),flush=True)
    print("ARCHIVE_LEDGER_BASE64 "+base64.b64encode(raw).decode(),flush=True)
if __name__=="__main__":main()
