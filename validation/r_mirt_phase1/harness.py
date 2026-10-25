"""Optional independent R mirt comparisons. No installation, simulation or app edits."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import inspect
import tempfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MODELS = ("1PL", "Rasch", "2PL", "3PL", "4PL", "GRM", "PCM", "GPCM", "NRM", "RSM")
SOURCES = {"1PL": "tests/fixtures/irt_b10/data_irt1pl_ok.csv",
    "Rasch": "tests/fixtures/irt_b10/data_irt1pl_ok.csv",
    "2PL": "tests/fixtures/irt_b10/data_irt2pl_ok.csv",
    "3PL": "tests/fixtures/irt_b11/data_irt3pl_ok.csv", "4PL": "tests/fixtures/irt_b11/data_irt4pl_ok.csv",
    **{m: "validation/evidence/p5_browser/poly_ordinal.csv" for m in ("GRM", "PCM", "GPCM")},
    "NRM": "validation/evidence/p5_browser/poly_nominal.csv", "RSM": "validation/evidence/p5_browser/rsm.csv"}


def discrepancies(a, b, tolerance=None):
    """Symmetric relative errors; zero/zero=0, zero/nonzero=1. No default cutoff."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(status="FAIL", reason="Nonfinite or incompatible arrays")
    delta = abs(a-b); scale = np.maximum(abs(a), abs(b))
    relative = np.divide(delta, scale, out=np.zeros_like(delta), where=scale > 0)
    status = "SKIPPED" if tolerance is None else "PASS" if np.all(delta <= tolerance) else "FAIL"
    return dict(status=status, max_absolute=float(delta.max()), mean_absolute=float(delta.mean()),
                max_relative=float(relative.max()), tolerance=tolerance,
                reason="Tolerance requires scientific calibration" if tolerance is None else "")


def semantic_parameters(prob, theta, model):
    """Common D=1 physical or baseline-category coordinates; no ordering of nominal codes."""
    prob = np.asarray(prob, float)
    if not np.isfinite(prob).all() or np.any(prob <= 0) or np.any(prob >= 1): raise ValueError("Degenerate probability trace")
    if model in ("1PL", "Rasch", "2PL", "3PL", "4PL"):
        return None  # Asymptotes require engine-native parameters, not grid endpoint guesses.
    if model == "GRM":
        survival = np.flip(np.cumsum(np.flip(prob, axis=1), axis=1), axis=1)[:, 1:]
        logits = np.log(survival / (1-survival))
    elif model == "NRM": logits = np.log(prob[:, 1:] / prob[:, :1])
    else: logits = np.log(prob[:, 1:] / prob[:, :-1])
    slopes = (logits[-1] - logits[0]) / (theta[-1] - theta[0])
    intercepts = logits[0] - slopes * theta[0]
    if model == "NRM": return dict(A=slopes.tolist(), D=intercepts.tolist())
    if np.any(abs(slopes) < 1e-12): raise ValueError("Unidentified step/threshold scale")
    return dict(a=slopes.tolist(), b=(-intercepts/slopes).tolist())


def python_fit(data, model, matched=False):
    theta = np.linspace(-4,4,81)
    if model == "Rasch":
        from core.rasch import fit_rasch
        from scipy.special import expit
        f = fit_rasch(data, points=61)
        p = expit(theta[:, None] - f["b"])
        probs = np.stack([1-p, p], axis=2); info = p*(1-p)
        meta = dict(converged=f["converged"], loglik=f["loglik"], aic=f["aic"], bic=f["bic"], variance=f["sd"]**2)
        params = [dict(a=1., b=float(b), c=0., d=1.) for b in f["b"]]
    elif model == "RSM":
        from core.irt_rsm import fit, item_moments
        f = fit(data, points=61)
        probs, _, info, _, _ = item_moments(theta, f["steps"])
        meta = dict(converged=f["converged"], loglik=f["loglik"], aic=f["aic"], bic=f["bic"], variance=f["sd"]**2)
        params = [semantic_parameters(probs[:,i],theta,model) for i in range(len(data.columns))]
    else:
        from core.irt_python_mirt_adapter import fit_python_mirt
        if matched:
            import mirt
            from mirt.estimation.base import _ITEM_PARAMETER_BOUNDS
            assert {k:_ITEM_PARAMETER_BOUNDS[k] for k in ("discrimination","difficulty","guessing","upper")} == dict(discrimination=(.1,5.),difficulty=(-6.,6.),guessing=(0.,.5),upper=(.5,1.))
            k = len(data.columns)
            start = dict(discrimination=np.ones(k),difficulty=np.zeros(k),guessing=np.full(k,.2))
            if model=="4PL": start["upper"]=np.full(k,.95)
            f = mirt.fit_mirt(data.to_numpy(int),model=model,n_quadpts=61,max_iter=3000,
                tol=1e-8,priors=None,accelerate="none",start_values=start,compute_standard_errors=False)
        else:
            f = fit_python_mirt(data, model=model, estimation="ML", n_categories=int(data.to_numpy().max())+1 if model in MODELS[5:9] else None).fit
        orientation = 1
        if model == "NRM":
            probe = f.model.probability(np.array([0.,1.]))[:,0]
            if np.log(probe[1,1]/probe[1,0])-np.log(probe[0,1]/probe[0,0]) < 0: orientation = -1
        probs = f.model.probability(theta * orientation)
        if model in MODELS[:5]: probs = np.stack([1-probs,probs],axis=2)
        info = np.column_stack([f.model.information(theta * orientation,item_idx=i) for i in range(len(data.columns))])
        meta = dict(converged=bool(f.converged), loglik=float(f.log_likelihood), aic=float(f.aic), bic=float(f.bic), variance=1.)
        if matched: meta["iterations"]=getattr(f,"n_iterations",None)
        if model in MODELS[:5]:
            from core.irt_python_mirt_parameters import python_mirt_binary_parameters
            params = python_mirt_binary_parameters(f,model=model)[["a","b","c","d"]].to_dict("records")
        else: params = [semantic_parameters(probs[:,i],theta,model) for i in range(len(data.columns))]
    rows = [dict(item=name, node=q, theta=t, category=c, probability=float(probs[q,i,c]), information=float(info[q,i]))
            for i,name in enumerate(data.columns) for q,t in enumerate(theta) for c in range(probs.shape[2])]
    return pd.DataFrame(rows), meta, params


def compare_curves(py, reference):
    keys = ["item","node","category"]
    if py.duplicated(keys).any() or reference.duplicated(keys).any(): return dict(status="FAIL",reason="Duplicate trace keys")
    pair = py.merge(reference, on=keys, suffixes=("_python","_R"), how="outer", indicator=True)
    if not pair._merge.eq("both").all(): return dict(status="FAIL",reason="Unmatched item/theta/category keys")
    result = {"probabilities": discrepancies(pair.probability_python,pair.probability_R),
              "theta_grid":discrepancies(pair.theta_python,pair.theta_R)}
    tif = pair.drop_duplicates(["item","node"]).groupby("node")[["information_python","information_R"]].sum()
    result["test_information"] = discrepancies(tif.information_python,tif.information_R)
    return result


def parameter_frame(names, parameters):
    return pd.DataFrame([dict(item=name, parameter=key, component=j, value=float(value))
        for name, p in zip(names, parameters) for key, v in p.items()
        for j, value in enumerate(np.atleast_1d(v))])


def compare_parameters(py, reference):
    keys = ["item","parameter","component"]
    if py.duplicated(keys).any() or reference.duplicated(keys).any(): return dict(status="FAIL",reason="Duplicate parameter keys")
    pair=py.merge(reference,on=keys,suffixes=("_python","_R"),how="outer",indicator=True)
    if not pair._merge.eq("both").all(): return dict(status="NOT COMPARABLE",reason="Parameter coordinates differ")
    return discrepancies(pair.value_python,pair.value_R)


def run(output, models=("1PL","2PL"), estimation="ML", rscript=None, matched=False):
    if matched and (estimation!="ML" or any(m not in ("3PL","4PL") for m in models)):
        raise ValueError("Matched profile is ML 3PL/4PL only")
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    executable = rscript if rscript is not None else shutil.which("Rscript")
    records = []
    for model in models:
        if model not in MODELS: raise ValueError("Unknown model")
        folder = output/model; folder.mkdir(exist_ok=True)
        source = ROOT/SOURCES[model]; data = pd.read_csv(source)
        category_map = {"A":0,"B":1,"C":2,"D":3} if model=="NRM" else None
        if category_map:
            if not data.isin(list(category_map)).all().all(): raise ValueError("Undeclared nominal label")
            data = data.apply(lambda col:col.map(category_map))  # Explicit labels, never success order.
        # Committed numeric item-only fixtures, not user records; no recoding/imputation.
        if not all(pd.api.types.is_numeric_dtype(d) for d in data.dtypes) or data.isna().any().any(): raise ValueError("Invalid fixture")
        path = folder/"data.csv"; data.to_csv(path,index=False)
        rec = dict(model=model, source=SOURCES[model], source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), shape=list(data.shape),
                   status="SKIPPED", execution="PENDING", settings=dict(D=1, estimation=estimation, seed=20261024,
                   category_map=category_map,
                   latent="N(0,sigma²), mean0/slope1/sigma free" if model in ("Rasch","RSM") else "N(0,1)",
                   scoring="unchanged 0..C-1; NRM labels unordered", R_quadrature=61, R_tol=1e-8),
                   se_ci=dict(status="NOT COMPARABLE",reason="Full covariance transformation and ML/MAP equivalence not yet audited"))
        records.append(rec)
        if estimation != "ML":
            rec.update(status="NOT COMPARABLE",execution="SKIPPED",reason="R/Python MAP priors and densities/Jacobians not matched"); continue
        if not executable:
            rec["reason"] = "Rscript unavailable; no R execution and no Python recalibration"; continue
        try:
            # Fresh outputs prevent a successful-looking stale R file from a prior run.
            reference_dir=Path(tempfile.mkdtemp(prefix="R_",dir=folder)).resolve()
            rec["reference_directory"]=reference_dir.name
            if matched:
                from scipy.special import roots_hermite
                nodes, weights = roots_hermite(61)
                pd.DataFrame(dict(theta=nodes*np.sqrt(2),weight=weights/np.sqrt(np.pi))).to_csv(reference_dir/"integration.csv",index=False)
                rec["settings"].update(profile="matched",Python_quadrature=61,Python_tol=1e-8,max_iter=3000,
                    bounds=dict(a=[.1,5],b=[-6,6],c=[0,.5],d=[.5,1]),priors=None,
                    stopping="Python absolute logLik increment; R parameter increment. Same numeric tol; criteria NOT equivalent.")
            completed = subprocess.run([executable,str(Path(__file__).with_name("reference.R")),model,str(path.resolve()),str(reference_dir)]+(["matched"] if matched else []),capture_output=True,text=True,timeout=1800 if matched else 180)
            (folder/"R_stdout.txt").write_text(completed.stdout); (folder/"R_stderr.txt").write_text(completed.stderr)
            if completed.returncode == 42:
                rec["reason"] = "R mirt package unavailable; installation not attempted"; continue
            rec["execution"] = "EXECUTED"
            if completed.returncode: rec.update(status="FAIL",reason="R script failed; inspect recorded stderr"); continue
            ref = pd.read_csv(reference_dir/"curves.csv"); rm = pd.read_csv(reference_dir/"metadata.csv").iloc[0].to_dict()
            if rm.get("engine")!="R mirt" or rm.get("model")!=model or rm.get("D")!=1 or rm.get("estimation")!="ML":
                raise ValueError("R reference engine/model/scale/estimator mismatch")
            if not np.isfinite([rm[k] for k in ("loglik","aic","bic","variance")]).all(): raise ValueError("Nonfinite R diagnostics")
            if matched and not rm.get("matched_profile"):
                raise ValueError("R reference did not execute matched profile")
            py, pm, params = python_fit(data,model,matched=True) if matched else python_fit(data,model); py.to_csv(folder/"python_curves.csv",index=False)
            if not np.isfinite([pm[k] for k in ("loglik","aic","bic","variance")]).all(): raise ValueError("Nonfinite Python diagnostics")
            rec.update(R=rm, python=pm, python_parameters=params, comparisons=compare_curves(py,ref), reason="Discrepancies recorded; acceptance tolerances require calibration")
            pf=parameter_frame(data.columns,params);pf.to_csv(folder/"python_parameters.csv",index=False)
            rec["comparisons"]["item_parameters"] = compare_parameters(pf,pd.read_csv(reference_dir/"parameters.csv"))
            rec["comparisons"]["diagnostics"] = {key:discrepancies([pm[key]],[rm[key]]) for key in ("loglik","aic","bic","variance")}
            if any(v.get("status")=="FAIL" for v in rec["comparisons"].values() if isinstance(v,dict)):
                rec.update(status="FAIL",reason="Invalid numerical comparison")
            if matched:
                from validation.r_mirt_phase1.equivalence_audit import evaluate, BOUNDS
                grid=pd.read_csv(reference_dir/"integration.csv")
                rp=pd.read_csv(reference_dir/"parameters.csv").pivot(index="item",columns="parameter",values="value").loc[data.columns]
                pp=pd.DataFrame(params,index=data.columns)
                common={key:evaluate(data.to_numpy(),p,grid.theta.to_numpy(),grid.weight.to_numpy())[2] for key,p in (("python",pp),("R",rp))}
                rec["comparisons"]["common_grid_loglik"]=discrepancies([common["python"]],[common["R"]])
                rec["common_grid_loglik"]=common
                rec["comparisons"]["parameters_by_name"]={k:discrepancies(pp[k],rp[k]) for k in BOUNDS}
                rec["boundary_flags"]={key:[dict(item=str(item),parameter=k,value=float(v),outside=bool(v<lo or v>hi))
                    for k,(lo,hi) in BOUNDS.items() if k!="d" or model=="4PL"
                    for item,v in p[k].items() if v<lo or v>hi or np.isclose(v,lo,atol=1e-6,rtol=0) or np.isclose(v,hi,atol=1e-6,rtol=0)] for key,p in (("python",pp),("R",rp))}
                if any(f["outside"] for fs in rec["boundary_flags"].values() for f in fs):
                    rec.update(status="FAIL",reason="Fitted parameters outside matched physical bounds")
            if not rm["converged"] or not pm["converged"]: rec.update(status="FAIL",reason="Nonconverged calibration")
        except subprocess.TimeoutExpired as exc:
            for name, value in (("R_stdout.txt",exc.stdout),("R_stderr.txt",exc.stderr)):
                (folder/name).write_text(value.decode(errors="replace") if isinstance(value,bytes) else value or "")
            rec.update(status="FAIL",execution="EXECUTED",reason="R process timed out; partial logs preserved")
        except (OSError, ValueError, KeyError) as exc:
            rec.update(status="FAIL",reason=f"{type(exc).__name__}: {exc}")
    import mirt
    report = dict(status="FAIL" if any(r["status"]=="FAIL" for r in records) else "PENDING" if any(r["execution"]=="PENDING" for r in records) else "NOT COMPARABLE" if all(r["status"]=="NOT COMPARABLE" for r in records) else "MEASURED_NOT_CERTIFIED",
        Rscript=executable,
        reference_script_sha256=hashlib.sha256(Path(__file__).with_name("reference.R").read_bytes()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        matched_source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("matched_ml.R","physical_b_constraint.R")} if matched else {},
        python_public_fit_signature=str(inspect.signature(mirt.fit_mirt)),
        python=sys.version.split()[0], versions={n:importlib.metadata.version(n) for n in ("numpy","scipy","pandas","mirt")},
        models=records, notice="Only actual R executions count as independent comparisons. No default PASS tolerance. Existing historical R fixtures are not new executions.")
    (output/"report.json").write_text(json.dumps(report,indent=2,default=str,allow_nan=False))
    return report


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    p.add_argument("--models",nargs="+",choices=MODELS,default=["1PL","2PL"])
    p.add_argument("--estimation",choices=["ML","MAP"],default="ML")
    p.add_argument("--matched",action="store_true")
    a=p.parse_args();r=run(a.output,a.models,a.estimation,matched=a.matched)
    print(json.dumps({"status":r["status"],"models":[{k:m[k] for k in ("model","status","execution")} for m in r["models"]]}))
