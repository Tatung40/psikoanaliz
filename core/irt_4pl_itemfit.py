"""Scoped 4PL standardized residual Q; native S-X2/ordinary chi-square stay descriptive."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd

METHOD_VERSION="4pl-standardized-gaussian-v1"
P_COLUMN="4PL Q kısıtlı referans p"
Q_COLUMN="4PL standartlaştırılmış artık Q"
STATUS_COLUMN="4PL Q bilimsel durumu"
METHOD="Ayrı standartlaştırılmış artık Q / kısıtlı Gauss MAP referansı"
SCOPE=("Yalnız N=3000/J=12, mirt 1.2.0 MAP/GH61/tol1e-4/max500, varsayılan önseller, "
       "kayıtlı madde tasarımı ve N(0,1) referans tasarımında doğrulandı. Kullanıcı DGP'si doğrulanmadı. "
       "Kabul: 24 madde-model için eşzamanlı Tip-I üst sınırı ≤.07; tam nominal .05 veya FWER/FDR sertifikası değildir.")
ROOT=Path(__file__).resolve().parents[1]
PROOF=ROOT/"validation/evidence/p3_4pl_closure/CERTIFICATION.json"


def _eligible(outcome,x):
    if getattr(outcome,"model",None)!="4PL" or getattr(outcome,"estimation",None)!="MAP":return False
    if getattr(outcome,"engine",None)!="python_mirt":return False
    if (getattr(outcome,"n_persons",None),getattr(outcome,"n_items",None))!=(3000,12):return False
    if getattr(outcome,"solver_settings",{})!=dict(max_iter=500,tol=1e-4,n_quadpts=61):return False
    if x.shape!=(3000,12) or not np.isfinite(x).all() or not np.isin(x,[0,1]).all():return False
    if not bool(getattr(outcome.fit,"converged",False)):return False
    return importlib.metadata.version("mirt")=="1.2.0"


def cache_fingerprint():
    """Proof/source withdrawal invalidates UI AND bundled reports without any refit."""
    digest=hashlib.sha256()
    try:raw=PROOF.read_bytes();proof=json.loads(raw)
    except (OSError,ValueError):return "missing-or-invalid-proof"
    digest.update(raw)
    for relative in sorted(proof.get("scientific_source_blobs",{})):
        digest.update(relative.encode())
        try:digest.update((ROOT/relative).read_bytes())
        except OSError:digest.update(b"MISSING")
    return digest.hexdigest()


def reference_table(outcome,responses,seed):
    """Calibration responses only: (new columns, refusal reason). No fits/p overwrite."""
    x=np.asarray(responses)
    if not _eligible(outcome,x):return None,"4PL Q referansı doğrulanmış MAP/N3000/J12 çözücü kapsamının dışında."
    try:proof=json.loads(PROOF.read_text())
    except (OSError,ValueError):return None,"Kısıtlı referans bağımsız doğrulama kaydı yok; çıkarım kapalı."
    if (proof.get("method_version")!=METHOD_VERSION or proof.get("status")!="SCOPED CERTIFIED"
            or not proof.get("independent_audit_passed") or not all(proof.get("gates",{}).values()) or not proof.get("gates")):
        return None,"Kısıtlı referans bilimsel kabul kapıları geçmedi; çıkarım kapalı."
    for relative,expected in proof.get("scientific_source_blobs",{}).items():
        try:raw=(ROOT/relative).read_bytes()
        except OSError:return None,"Kısıtlı referans hesaplama kaynağı yok; çıkarım kapalı."
        if hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()!=expected:
            return None,"Kısıtlı referans hesaplama kaynağı değişti; yeniden doğrulama gerekli."
    if not proof.get("scientific_source_blobs") or not proof.get("certified_claims"):return None,"4PL Q kaynak veya kapsam kanıtı eksik."
    if seed is None:raise ValueError("Tekrarlanabilir bağımsız referans seed'i gerekli.")
    import mirt
    # Reuse the frozen, independently confirmed kernel. Its exact sources ship in
    # reproducibility ZIPs and are hash-checked; it never recalibrates an item model.
    from validation.p3_empirical.constrained import analyse
    eta=outcome.fit.coef()[["discrimination","difficulty","guessing","upper"]].to_numpy(float).ravel()
    try:
        result=analyse(x,eta,"4PL",seed)
        native=mirt.itemfit(outcome.fit,x,statistics=["S_X2"])
        identity=float(np.max(abs(native.S_X2.to_numpy()-np.array([v["sx"] for v in result["items"]]))))
        if not np.isfinite(identity) or identity>1e-8:raise ValueError("S-X² değer eşdeğerliği doğrulanmadı.")
        if result["diagnostics"]["max_KKT"]>1e-7:raise ValueError("Kısıtlı referans KKT denetimi başarısız.")
    except (ValueError,np.linalg.LinAlgError) as exc:
        return None,"Kısıtlı referans sayısal ret; p üretilmedi: "+str(exc)
    p=np.array([v["p_constrained_standardized"] for v in result["items"]])
    q=np.array([v["Q_constrained"] for v in result["items"]])
    valid=np.isfinite(native.p_value.to_numpy())&np.isfinite(p)&np.isfinite(q)&(p>=0)&(p<=1)
    p=np.where(valid,p,np.nan)
    frame=pd.DataFrame({Q_COLUMN:np.where(valid,q,np.nan),P_COLUMN:p,"Karar (4PL Q, α=.05)":np.where(~valid,"Hesaplanamadı",np.where(p<.05,"Uyumsuz","Reddedilmedi")),
        "4PL Q yöntemi":METHOD,"Referans B":4095,
        "Referans seed":str(seed),STATUS_COLUMN:np.where(valid,"SCOPED CERTIFIED — referans tasarımı","ÇIKARIM REDDEDİLDİ"),
        "4PL Q kapsamı":SCOPE+" Güç doğrulaması: "+", ".join({"nonmonotone":"nonmonotonluk","local_dependence":"yerel bağımlılık"}.get(k,k) for k in proof["certified_claims"])+". İki aşamalı şekil için duyarlılık doğrulanmadı."})
    frame.attrs["p3_4pl_reference"]={"method_version":METHOD_VERSION,"seed":seed,"identity_max":identity,
        "seconds":result["seconds"],"ordinary_chi_square_p":"NOT CERTIFIED","reference":proof.get("confirmation_run")}
    return frame,None
