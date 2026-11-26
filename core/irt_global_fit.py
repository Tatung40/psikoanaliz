"""Native mirt binary global diagnostics; ML theory and scoped MAP references stay separate."""
import hashlib,json,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2
ROOT=Path(__file__).resolve().parents[1]
PROOF=ROOT/"validation/evidence/binary_global_fit/CERTIFICATION.json"
VERSION="binary-global-fit-O1"

def cache_fingerprint():
    h=hashlib.sha256()
    try:
        raw=PROOF.read_bytes();h.update(raw);proof=json.loads(raw)
        for p in sorted(proof.get("scientific_source_blobs",{})):
            h.update(p.encode());h.update((ROOT/p).read_bytes())
    except (OSError,ValueError):h.update(b"MISSING")
    return h.hexdigest()

def _coordinates(outcome):
    co=outcome.fit.coef();J=len(co);q=int(outcome.fit.n_parameters)
    a=co["discrimination"].to_numpy(float) if "discrimination" in co else np.ones(J)
    b=co["difficulty"].to_numpy(float)
    model=outcome.model
    if model in ("Rasch","1PL") and q==J and np.allclose(a,1,rtol=0,atol=1e-12):return "Rasch",b
    if model=="1PL" and q==J+1 and np.allclose(a,a[0],rtol=0,atol=1e-12):return "common1PL",np.r_[b,a[0]]
    if model not in ("2PL","3PL","4PL"):raise ValueError("Unsupported fixed/common discrimination identification.")
    cols=["discrimination","difficulty"]+(["guessing"] if model in ("3PL","4PL") else [])+(["upper"] if model=="4PL" else [])
    if q!=J*len(cols):raise ValueError("Free parameter layout does not match model.")
    return model,co[cols].to_numpy(float).ravel()

def _proof():
    p=json.loads(PROOF.read_text())
    if p.get("version")!=VERSION:raise ValueError("Global reference version mismatch.")
    for path,expected in p.get("scientific_source_blobs",{}).items():
        b=(ROOT/path).read_bytes()
        if hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()!=expected:raise ValueError("Global reference sources changed.")
    if not p.get("scientific_source_blobs"):raise ValueError("Missing reference source identity.")
    return p

def global_fit(outcome,responses,*,seed=None):
    """Same complete calibration sample; no estimator, person-score conditioning or hidden deletion."""
    if getattr(outcome,"engine",None)!="python_mirt" or outcome.model not in ("Rasch","1PL","2PL","3PL","4PL"):
        raise ValueError("Native unidimensional binary mirt outcome required.")
    x=np.asarray(responses,float);N,J=x.shape
    if not 3<=J<=30 or N<30 or not np.isfinite(x).all() or not np.isin(x,[0,1]).all():raise ValueError("Complete binary N>=30/J3..30 required.")
    if (outcome.n_persons,outcome.n_items)!=(N,J) or int(outcome.fit.n_observations)!=N:raise ValueError("Calibration cohort dimensions differ.")
    if not bool(outcome.fit.converged):raise ValueError("No global inference for nonconvergence.")
    if importlib.metadata.version("mirt")!="1.2.0":raise ValueError("Global diagnostic API validated for mirt1.2.0.")
    mode,eta=_coordinates(outcome)
    model=outcome.model
    cons=[{"parameter":"discrimination","items":list(range(J))}] if mode=="common1PL" else None
    from mirt.diagnostics.modelfit import compute_fit_indices
    from validation.binary_global_fit import kernel as K
    points=int(outcome.solver_settings.get("n_quadpts",61))
    native=compute_fit_indices(outcome.fit,x,n_quadpts=points,prior_mean=[0.],prior_cov=[[1.]],constraints=cons)
    obs=K.observed(x);mu,Xi,D=K.moments(eta,mode,J,points=points)
    m2,rank,_,_=K.m2_from_moments(obs,mu,Xi,D,N,rank_threshold=1e-6)
    if rank["df"]!=native["M2_df"] or abs(m2-native["M2"])>1e-7+1e-6*abs(native["M2"]):raise ValueError("Native/independent M2 rank or statistic mismatch.")
    sr,res=K.srmsr(obs,mu,J);baseline,base_rank=K.baseline(obs,J,N)
    if abs(sr-float(native["SRMSR"]))>1e-7:raise ValueError("Native SRMSR identity mismatch.")
    status="NUMERICAL AGREEMENT / DESCRIPTIVE"
    reason="MAP M2 χ² and RMSEA intervals are not certified; descriptive indices do not imply fit."
    p_asym=np.nan
    co=outcome.fit.coef();bound=False
    for name,lo,hi in (("discrimination",.1,5.),("difficulty",-6.,6.)):
        if name in co:bound=bound or bool(((co[name]-lo<=1e-6)|(hi-co[name]<=1e-6)).any())
    if outcome.estimation=="ML" and mode in ("Rasch","common1PL","2PL") and rank["rank_stable"] and rank["rank"]==rank["q_nominal"] and not bound:
        p_asym=float(native["M2_p"]);status="ASYMPTOTIC ML / NUMERICALLY VERIFIED / NOT MC CERTIFIED"
        reason="Regular interior fixed-scale ML M2 theory; no finite-sample simulation certificate is claimed."
    row={"Model":model,"Kestirim":outcome.estimation,"N":N,"Madde":J,
         "M2":float(native["M2"]),"M2 etkin sd":int(native["M2_df"]),"M2 asimptotik p":p_asym,
         "RMSEA2 (N−1)":float(native["RMSEA"]),"SRMSR":float(native["SRMSR"]),
         "CFI":float(native["CFI"]),"TLI":float(native["TLI"]),
         "Bağımsızlık M2":baseline,"Bağımsızlık sd":base_rank["df"],
         "Jacobian rank":rank["rank"],"Nominal parametre":rank["q_nominal"],"Rank kararlı":rank["rank_stable"],
         "QLI (ayrı istatistik)":np.nan,"Küresel referans p":np.nan,"Küresel yöntem":"Native M2; moment-GLS, numeric1e-6 tangent rank",
         "Bilimsel durum":status,"Sınırlılık":reason,"RMSEA2 GA":"R ve Python N/N−1/fallback kuralları farklı; doğrulanmadı, gösterilmez"}
    if baseline<=base_rank["df"]:
        row["CFI"]=row["TLI"]=np.nan
        row["Sınırlılık"]+=" Independence baseline has no positive noncentrality; perfect-fit fallback not imputed."
    eligible=(model in ("3PL","4PL") and outcome.estimation=="MAP" and (N,J)==(3000,12)
              and outcome.solver_settings==dict(max_iter=500,tol=1e-4,n_quadpts=61))
    if eligible:
        try:
            proof=_proof();cell=proof["models"][model]
            if cell["status"]!="SCOPED CERTIFIED" or not all(cell["gates"].values()) or not cell["certified_claims"]:raise ValueError("Global independent gates incomplete.")
            if seed is None:raise ValueError("Independent explicit global reference seed required.")
            from validation.binary_global_fit.analysis import analyse
            r=analyse(x.astype(int),eta,model,seed)
            p=float(r[cell["method"]])
            if not np.isfinite(p) or not 0<=p<=1:raise ValueError("Global reference numeric refusal.")
            row["Küresel referans p"]=p;row["Küresel yöntem"]=cell["method_label"]
            if cell["method"]=="p_QLI":row["QLI (ayrı istatistik)"]=r["QLI"]
            row["Bilimsel durum"]="SCOPED CERTIFIED — kayıtlı N3000/J12/default cold MAP/GH61/max500/tol1e-4"
            row["Sınırlılık"]="Kullanıcı DGP’si doğrulanmaz; sertifikalı duyarlılık: "+", ".join(cell["certified_claims"])+". FWER/FDR, diğer N/J ve D4 profili kapsam dışı."
            row["Küresel seed"]=str(seed)
        except (OSError,ValueError,KeyError,np.linalg.LinAlgError) as e:
            row["Sınırlılık"]+=" Scoped global inference withheld: "+str(e)
    return dict(table=pd.DataFrame([row]),residuals=pd.DataFrame(res,index=outcome.fit.coef().index,columns=outcome.fit.coef().index),rank=rank,scope_profile="MML moments, fixed latent N(0,1), complete binary",
                ordinary_MAP_p="WITHHELD",data_sha256=hashlib.sha256(x.astype(np.uint8).tobytes()).hexdigest())

def comparison_context(outcome,responses):
    """Actual native fit identity for regular nested-ML comparisons."""
    x=np.asarray(responses)
    if x.ndim!=2 or not np.isfinite(x).all() or not np.isin(x,[0,1]).all():raise ValueError("Same complete binary cohort required.")
    mode,eta=_coordinates(outcome);N,J=x.shape
    if getattr(outcome,"engine",None)!="python_mirt" or outcome.n_persons!=N:raise ValueError("Native cohort identity mismatch.")
    fit=outcome.fit;ll=float(fit.log_likelihood);q=int(fit.n_parameters)
    if not np.isfinite(ll):raise ValueError("Finite likelihood required.")
    from validation.binary_global_fit import kernel as K
    mu,Xi,D=K.moments(eta,mode,J,points=int(outcome.solver_settings["n_quadpts"]))
    _,_,rank=K.projection(Xi,D,rank_threshold=1e-6)
    a,b,c,d=K.unpack(eta,mode,J)
    boundary=bool(np.any(abs(b)>=6-1e-6) or np.any(a<=.1+1e-6) or np.any(a>=5-1e-6))
    return dict(model=outcome.model,family=mode,N=N,J=J,objective=outcome.estimation,ll=ll,q=q,
        converged=bool(fit.converged),boundary=boundary,regular=rank["rank_stable"] and rank["rank"]==rank["q_nominal"],
        settings=dict(outcome.solver_settings),latent="N(0,1) fixed",data_sha256=hashlib.sha256(x.astype(np.uint8).tobytes()).hexdigest())

def nested_lrt(lower,upper):
    """Regular ML Wilks only; MAP and guessing/upper-asymptote boundaries fail closed."""
    row=dict(Alt=lower["model"],Üst=upper["model"],LR=np.nan,sd=np.nan,p=np.nan,**{"2ΔlogL (betimsel)":2*(upper["ll"]-lower["ll"])},
             Durum="NOT AVAILABLE",Gerekçe="")
    checks=[(lower["data_sha256"]==upper["data_sha256"] and (lower["N"],lower["J"])==(upper["N"],upper["J"]),"Cohort differs"),
            (lower["settings"]==upper["settings"] and lower["latent"]==upper["latent"],"Integration/scale differs"),
            (lower["objective"]==upper["objective"]=="ML","MAP is not an ML likelihood-ratio test"),
            (lower["converged"] and upper["converged"],"Nonconvergence"),
            (not lower["boundary"] and not upper["boundary"] and lower["regular"] and upper["regular"],"Boundary/rank/identification obstruction"),
            ((lower["family"],upper["family"]) in (("Rasch","common1PL"),("Rasch","2PL"),("common1PL","2PL")),"Unsupported nesting or c=0/d=1 boundary; no ordinary Wilks")]
    for ok,reason in checks:
        if not ok:row["Gerekçe"]=reason;return row
    stat=row["2ΔlogL (betimsel)"];df=upper["q"]-lower["q"]
    if stat<0 or df<=0:row["Gerekçe"]="Negative LR/nonpositive df: no clipping; fits not ordered";return row
    row.update(LR=stat,sd=df,p=float(chi2.sf(stat,df)),Durum="ASYMPTOTIC ML / NOT MC CERTIFIED",Gerekçe="Regular interior nested models; finite-sample size not certified.")
    return row
