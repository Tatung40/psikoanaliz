"""Transport-only lossless selected-method ledger; no new statistical experiment."""
import argparse,base64,gzip,hashlib,json,math
from pathlib import Path
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",required=True);ap.add_argument("--out",required=True);z=ap.parse_args()
    src=Path(z.input);report=json.loads((src/"AUDIT.json").read_text());raw=(src/"CONFIRMATION.jsonl.gz").read_bytes()
    assert hashlib.sha256(raw).hexdigest()==report["archive_sha256"]
    rows=[json.loads(s) for s in gzip.decompress(raw).splitlines()]
    compact=[]
    for r in rows:
        row={k:r[k] for k in ("model","kind","root","r","seed","data_sha256","source","run","converged","iterations","mirt","python","seconds") if k in r}
        if r.get("error"):row["error"]=r["error"]
        if r.get("result"):
            result=r["result"];items=result["items"]
            row.update(p_grid=[None if a["p_constrained_standardized"] is None else int(round(4096*a["p_constrained_standardized"])) for a in items],
                Q=[a["Q_constrained"] for a in items],sx=[a["sx"] for a in items],sx_native=r["sx_native"],
                native_valid=[v is not None and math.isfinite(v) for v in r["p_chi"]],
                identity=r["identity"],max_KKT=result["diagnostics"]["max_KKT"],kernel_seconds=result["seconds"])
        compact.append(row)
    blob=gzip.compress("".join(json.dumps(r,separators=(",",":"))+"\n" for r in compact).encode(),mtime=0)
    report["full_original_archive_sha256"]=report.pop("archive_sha256")
    report["full_original_archive_git_blob"]=report.pop("archive_git_blob")
    report["archive_sha256"]=hashlib.sha256(blob).hexdigest()
    report["archive_git_blob"]=hashlib.sha1(b"blob "+str(len(blob)).encode()+b"\0"+blob).hexdigest()
    report["ledger_schema"]="selected-Q-v1: p_grid/4096 is the EXACT p; Q/native/analysis S-X2 floats unchanged; all seeds, hashes, failures retained. Unselected statistics and full diagnostics remain original canonical artifacts."
    dest=Path(z.out);dest.mkdir(parents=True,exist_ok=True)
    (dest/"CONFIRMATION.jsonl.gz").write_bytes(blob);(dest/"AUDIT.json").write_text(json.dumps(report,indent=1))
    print("INDEPENDENT_AUDIT_JSON "+json.dumps(report),flush=True)
    text=base64.b64encode(blob).decode();print("ARCHIVE_META "+json.dumps(dict(bytes=len(blob),base64=len(text))),flush=True)
    for i in range(0,len(text),100000):print("ARCHIVE_CHUNK "+str(i//100000)+" "+text[i:i+100000],flush=True)
if __name__=="__main__":main()
