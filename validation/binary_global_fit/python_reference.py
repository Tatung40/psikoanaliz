"""Actual native APIs at fixed parameters; NO calibration."""
import json,inspect
from pathlib import Path
import numpy as np,pandas as pd
import mirt
from mirt.diagnostics.modelfit import compute_fit_indices
from mirt.models.dichotomous import OneParameterLogistic,TwoParameterLogistic,ThreeParameterLogistic,FourParameterLogistic
from validation.binary_global_fit import kernel as K
def eta(model,J=12):
    a=np.linspace(.8,2,J);b=np.linspace(-1.25,1.25,J)
    if model=="Rasch":return b
    if model=="common1PL":return np.r_[b,1.2]
    return np.stack([a,b,np.full(J,.15),np.full(J,.85)][:{"2PL":2,"3PL":3,"4PL":4}[model]],axis=1).ravel()
def native_model(model,J,e):
    cls={"Rasch":OneParameterLogistic,"common1PL":TwoParameterLogistic,"2PL":TwoParameterLogistic,
         "3PL":ThreeParameterLogistic,"4PL":FourParameterLogistic}[model]
    print("ACTUAL_MODEL_API",cls.__name__,str(inspect.signature(cls)),flush=True)
    obj=cls(n_items=J);a,b,c,d=K.unpack(e,model,J)
    p=dict(difficulty=b)
    if model!="Rasch":p["discrimination"]=a
    if model in ("3PL","4PL"):p["guessing"]=c
    if model=="4PL":p["upper"]=d
    print("ACTUAL_PARAMETERS",list(obj.parameters),flush=True)
    obj.set_parameters(**p)
    cons=[{"parameter":"discrimination","items":list(range(J))}] if model=="common1PL" else None
    return obj,cons
def main():
    out=Path("/tmp/global-reference");out.mkdir(parents=True,exist_ok=True)
    x=pd.read_csv("tests/fixtures/irt_b11/data_irt3pl_ok.csv").iloc[:600].to_numpy()
    pd.DataFrame(x,columns=[f"I{i+1}" for i in range(12)]).to_csv(out/"data.csv",index=False)
    t,w=K.grid(61);np.savetxt(out/"GH61.csv",np.c_[t,w],delimiter=",",header="theta,weight",comments="",fmt="%.17g")
    records=[];params=[]
    for model in ("Rasch","common1PL","2PL","3PL","4PL"):
        e=eta(model);obj,cons=native_model(model,12,e)
        actual=mirt.compute_m2(obj,x,n_quadpts=61,constraints=cons)
        idx=compute_fit_indices(obj,x,n_quadpts=61,constraints=cons)
        mu,Xi,D=K.moments(e,model,12);obs=K.observed(x)
        math,diag,_,_=K.m2_from_moments(obs,mu,Xi,D,len(x),rank_threshold=1e-6)
        base,bd=K.baseline(obs,12,len(x))
        sr,_=K.srmsr(obs,mu,12)
        record=dict(model=model,native=actual,indices=idx,independent_M2=math,rank=diag,baseline_M2=base,baseline_df=bd["df"],mu=mu.tolist(),Xi=Xi.tolist(),D=D.tolist(),observed=obs.tolist(),
            SRMSR_independent=sr,probability_error=float(np.max(abs(obj.probability(t[:,None])-(K.unpack(e,model,12)[2]+(K.unpack(e,model,12)[3]-K.unpack(e,model,12)[2])*__import__("scipy").special.expit(K.unpack(e,model,12)[0]*(t[:,None]-K.unpack(e,model,12)[1])))))))
        error=abs(actual["M2"]-math);record["numerical_identity_pass"]=bool(actual["df"]==diag["df"] and error<=1e-7+1e-6*abs(math) and abs(idx["SRMSR"]-sr)<1e-7)
        records.append(record)
        a,b,c,d=K.unpack(e,model,12)
        for j in range(12):params.append(dict(model=model,item=j+1,a=a[j],b=b[j],c=c[j],d=d[j]))
    pd.DataFrame(params).to_csv(out/"parameters.csv",index=False)
    (out/"PYTHON.json").write_text(json.dumps(records,indent=1,default=lambda x:x.item()))
    print("NATIVE_REFERENCE_JSON "+json.dumps([{k:v for k,v in r.items() if k not in ("mu","Xi","D","observed")} for r in records],default=lambda x:x.item()),flush=True)
    assert all(r["numerical_identity_pass"] for r in records)
if __name__=="__main__":main()

# Prospective exact operating characteristics of the registered final gates.
from scipy.stats import binom
from statsmodels.stats.proportion import proportion_confint
n=6000
eligible=[k for k in range(n+1) if proportion_confint(k,n,alpha=.05,method="beta")[1]<=.07 and proportion_confint(k,n,alpha=.05,method="beta")[0]>=.03 and np.sqrt((k/n)*(1-k/n)/n)<=.004]
assert eligible
print("DESIGN_PRECISION_JSON "+json.dumps({"R":n,"accepted_count_range":[min(eligible),max(eligible)],"pass_probability_at_05":float(binom.cdf(max(eligible),n,.05)-binom.cdf(min(eligible)-1,n,.05))}))
