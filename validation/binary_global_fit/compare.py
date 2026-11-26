"""Strict matched-quantity reference audit; retain native-R numerical obstructions."""
import json,math
from pathlib import Path
import numpy as np
p=Path("/tmp/global-reference");P=json.loads((p/"PYTHON.json").read_text());R=json.loads((p/"R.json").read_text())
out=[]
for a in P:
    model=a["model"];b=R[model]
    errors=dict(mean=float(np.max(abs(np.array(a["mu"])-np.array(b["means"])))),covariance=float(np.max(abs(np.array(a["Xi"])-np.array(b["Xi"])))),
        Jacobian=float(np.max(abs(np.array(a["D"])-np.array(b["Jacob"])))),R_math_M2=abs(a["native"]["M2"]-b["independent_R_M2"]))
    good=errors["mean"]<=1e-10 and errors["covariance"]<=1e-10 and errors["Jacobian"]<=1e-7+1e-5*np.max(abs(np.array(a["D"])))
    good=good and a["native"]["df"]==b["independent_R_df"] and errors["R_math_M2"]<=1e-7+1e-6*abs(a["native"]["M2"])
    baseline_error=abs(a["baseline_M2"]-b["exact_independence"]["M2"])
    index_errors={k:abs(a["indices"][k]-b["independent_indices"][k]) for k in ("CFI","TLI")}
    good=good and baseline_error<=1e-7+1e-6*abs(a["baseline_M2"]) and all(v<=1e-7+1e-6*abs(b["independent_indices"][k]) for k,v in index_errors.items())
    same_df=a["native"]["df"]==b["mirt"]["df"]
    m2_error=abs(a["native"]["M2"]-b["mirt"]["M2"])
    reconstruct=abs(b["native_R_reconstructed"]-b["mirt"]["M2"])
    qr_obstruction=b["native_R_QR_rank"]<b["native_R_columns"] and reconstruct<=1e-7+1e-6*abs(b["mirt"]["M2"])
    derivative_obstruction=max(v["shape_error"] for v in b["native_dP_audit"])>1e-5 and reconstruct<=1e-7+1e-6*abs(b["mirt"]["M2"])
    if same_df and m2_error<=1e-7+1e-6*abs(a["native"]["M2"]):native_status="NUMERICAL AGREEMENT (M2, SRMSR, RMSEA); exact-baseline CFI/TLI verified separately"
    elif not same_df:native_status="NOT COMPARABLE: Python effective rank versus R nominal free parameter count"
    elif qr_obstruction:native_status="NOT COMPARABLE: independently reconstructed R QR rank loss; nominal df hides discarded directions"
    elif derivative_obstruction:native_status="NOT COMPARABLE: independently reconstructed native R dP disagrees with finite-difference ProbTrace derivative"
    else:native_status="UNRESOLVED";good=False
    out.append(dict(model=model,errors=errors,native_R_M2_error=m2_error,R_mirt_status=native_status,
        native_df=a["native"]["df"],R_mirt_df=b["mirt"]["df"],R_QR_rank=b["native_R_QR_rank"],R_QR_columns=b["native_R_columns"],
        R_reconstructed_error=reconstruct,R_dP_max_shape_error=max(v["shape_error"] for v in b["native_dP_audit"]),
        exact_baseline_error=baseline_error,matched_index_errors=index_errors,independent_mathematics_pass=bool(good)))
report=dict(reference="Actual R1.48 plus independent R FD/probability/moment math; failed native comparisons not relabeled agreement",models=out)
(p/"COMPARISON.json").write_text(json.dumps(report,indent=1))
print("GLOBAL_REFERENCE_JSON "+json.dumps(report),flush=True)
assert all(x["independent_mathematics_pass"] for x in out)
