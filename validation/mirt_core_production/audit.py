"""Archive/source audit only. Never calibrates or claims a new R execution."""
import argparse,ast,hashlib,json,platform
from pathlib import Path
from importlib.metadata import version
ROOT=Path(__file__).resolve().parents[2]
def blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
def audit():
    owner=json.loads((ROOT/"validation/mirt_core_production/OWNERSHIP.json").read_text())
    for entry in owner["preserve_unchanged"]:
        assert blob((ROOT/entry["path"]).read_bytes())==entry["git_blob"],entry["path"]
    source=(ROOT/"core/irt_multidim.py").read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and
              any(isinstance(t,ast.Name) and t.id=="BIFACTOR_RECOVERY" for t in n.targets))
    lines=source.splitlines(keepends=True)
    historical="".join(lines[:node.lineno-1]+lines[node.end_lineno:]).encode()
    assert blob(historical)==owner["canonical_source"]["core_blob"],"Canonical native source changed beyond warning"
    assert version("mirt")=="1.2.0"
    criteria=json.loads((ROOT/"validation/mirt_stage1/criteria.json").read_text())
    expected={"slopes":"calibration_parameters","intercepts":"calibration_parameters","covariance":"calibration_covariance",
              "probabilities":"calibration_probability","information":"calibration_information","EAP":"calibration_EAP",
              "posterior_SD":"calibration_posterior_SD","common_logLik":"common_logLik",
              "Python_native_LL":"native_common_logLik","R_native_LL":"native_common_logLik"}
    names={"2PL_2D","2PL_3D","GRM_2D","GRM_3D","GPCM_3D","PCM_2D","Rasch_2D"}
    records={}
    for path in ("phase1_evidence.json","expansion_original_evidence.json"):
        document=json.loads((ROOT/"validation/mirt_stage1"/path).read_text())
        for r in document["reports"]:
            if r["profile"] not in names:continue
            assert r["Python"]["converged"] is True and r["R"]["converged"] is True
            assert r["R"]["version"]=="1.48" and r["R_execution"]=="CALIBRATION EXECUTED"
            assert r["R"]["constraints_valid"] is True
            for key,criterion in expected.items():
                c=r["comparison"][key]
                assert c["status"]=="PASS" and c["tolerance"]==criteria[criterion]
                assert 0<=c["max_absolute"]<=criteria[criterion],(r["profile"],key)
            records[r["profile"]]={"run":document["run"],"scope":r["settings"],
                                    "differences":{k:r["comparison"][k]["max_absolute"] for k in expected}}
    assert set(records)==names
    from core.irt_mirt_policy import profiles
    assert set(profiles())==names|{"NRM_2D_COLD_C1"}
    cold=json.loads((ROOT/"validation/mirt_core_production/NRM_COLD_REFERENCE.json").read_text())
    c=cold["report"]
    assert cold["provenance"]["run_id"]==45960533214 and c["calibration_attempts"]=={"Python":1,"R":1}
    assert c["Python"]["converged"] is True and c["R"]["converged"] is True
    assert c["Python"]["admissibility"]["valid"] is True and c["R_admissibility"]["valid"] is True
    assert c["protocol"]["commit"]=="af024e73e4be99d0db1a2a0df400039f98ab6095"
    for key,criterion in expected.items():
        check=c["comparison"][key]
        assert check["status"]=="PASS" and check["tolerance"]==criteria[criterion]
        assert 0<=check["max_absolute"]<=criteria[criterion],key
    return dict(status="PASS",new_calibrations=0,new_R_calibrations=0,Python=platform.python_version(),
                mirt=version("mirt"),baseline=owner["baseline_main"],profiles=records,
                canonical_source_equivalence="Verified exact source after removing warning-only constant",
                archive_limit="Original raw artifact40499970808 unavailable; retained immutable numerical reports audited. No new saved-fit reevaluation claimed.",
                inference="WITHHELD; no new inference/coverage certificate")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("validation-output/mirt-core/audit.json"));args=p.parse_args()
    result=audit();args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+"\n")
    print("MIRT_CORE_ARCHIVE_AUDIT="+json.dumps({"status":result["status"],"profiles":len(result["profiles"]),"new_calibrations":0}))
