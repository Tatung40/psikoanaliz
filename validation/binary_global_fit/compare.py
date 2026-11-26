"""Independent R/Python fixed-point audit; no fits, no inflated R-authority claims."""
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
    same_df=a["native"]["df"]==b["mirt"]["df"]
    native_status="NOT COMPARABLE: native1e-6 rank versus R nominal free-parameter count"
    native_errors={}
    if same_df:
        for pykey,rkey in (("M2","M2"),("RMSEA","RMSEA"),("SRMSR","SRMSR"),("CFI","CFI"),("TLI","TLI")):
            if rkey in b["mirt"] and pykey in a["indices"]:
                x,y=a["indices"][pykey],b["mirt"][rkey]
                if x is not None and y is not None and math.isfinite(x) and math.isfinite(y):native_errors[pykey]=abs(x-y)
        good=good and all(v<=1e-7+1e-6*abs(b["mirt"][k]) for k,v in native_errors.items())
        native_status="NUMERICAL AGREEMENT" if good else "FAIL"
    out.append(dict(model=model,errors=errors,R_mirt_status=native_status,R_mirt_errors=native_errors,
                    native_df=a["native"]["df"],R_mirt_df=b["mirt"]["df"],independent_mathematics_pass=bool(good)))
report=dict(reference="Actual mirt1.48 fixed-point (not fitted) and independent R/Python moment math",models=out)
(p/"COMPARISON.json").write_text(json.dumps(report,indent=1))
print("GLOBAL_REFERENCE_JSON "+json.dumps(report),flush=True)
assert all(x["independent_mathematics_pass"] for x in out)
