"""Guarded multi-start 3PL/4PL calibration and reliability audit.

Multi-start is a numerical safeguard, not an identification remedy.
All candidates and failures are retained. A converged solution is not
automatically accepted as statistically reliable.
"""
import numpy as np
import pandas as pd
from core.irt import fit_irt
from core.irt_se_diagnostics import hessian_boundary_diagnostics

def fit_irt_multistart(data,model="3PL",starts=(0.,-1.,1.),**kwargs):
    if model not in ("3PL","4PL"):
        raise ValueError("Çoklu başlangıç 3PL/4PL için tasarlanmıştır.")
    if not starts or len(starts)>7 or not np.isfinite(starts).all():
        raise ValueError("Geçersiz başlangıç listesi.")
    if "initial_shift" in kwargs:
        raise ValueError("initial_shift yerine starts kullanın.")
    fits=[]; logs=[]
    for shift in starts:
        try:
            fit=fit_irt(data,model=model,initial_shift=float(shift),**kwargs)
            if not np.isfinite(fit["loglik"]):
                raise RuntimeError("Sonlu logLik yok.")
            diag=hessian_boundary_diagnostics(
                fit["items"],model,
                a_upper=fit["a_upper"],d_lower=fit["d_lower"])
            fits.append((float(shift),fit,diag))
            logs.append({"Başlangıç":float(shift),"Başarılı":True,
                         "Yakınsama":bool(fit["converged"]),
                         "logLik":float(fit["loglik"]),
                         "Sınırdaki parametre":len(diag),
                         "Yineleme":int(fit["iterations"]),"Hata":""})
        except (RuntimeError, ValueError, FloatingPointError,
                np.linalg.LinAlgError) as exc:
            logs.append({"Başlangıç":float(shift),"Başarılı":False,
                         "Yakınsama":False,"logLik":np.nan,
                         "Sınırdaki parametre":np.nan,
                         "Yineleme":np.nan,"Hata":str(exc)})
    if not fits:
        raise RuntimeError("Hiçbir başlangıçta kestirim üretilemedi.")
    # Prefer converged candidate; do not silently treat nonconvergence as success.
    converged=[entry for entry in fits if entry[1]["converged"]]
    selected=max(converged or fits,key=lambda entry:entry[1]["loglik"])
    shift,best,diagnostics=selected
    table=pd.DataFrame(logs)
    best_loglik=float(best["loglik"])
    near=table[(table["Yakınsama"])&(table["logLik"]>=best_loglik-.01)]
    spread=float(near["logLik"].max()-near["logLik"].min()) if len(near)>1 else np.nan
    summary={
        "Başlangıç sayısı":len(starts),
        "Başarılı kestirim":len(fits),
        "Yakınsayan":len(converged),
        "Seçilen başlangıç":shift,
        "Seçilen logLik":best_loglik,
        "Sınırdaki parametre":len(diagnostics),
        "Sınırdaki madde":int(diagnostics["Madde"].nunique()),
        "Eşdeğer logLik aday sayısı":int(len(near)),
        "Eşdeğer logLik aralığı":spread,
        "Not":"Yakınsama ve yüksek logLik tek başına parametre "
              "tanımlanabilirliğini veya SH geçerliğini kanıtlamaz."
    }
    return {"fit":best,"candidates":table,"boundary":diagnostics,
            "summary":summary}
