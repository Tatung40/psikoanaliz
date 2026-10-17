"""Conservative, descriptive 3PL/4PL calibration reliability assessment.

Not a statistical goodness-of-fit test or evidence of construct validity.
Never labels a fit scientifically validated solely because EM converged.
"""
from core.irt_se_diagnostics import hessian_boundary_diagnostics


def model_reliability(fit):
    model=fit["model"]
    if model not in ("3PL","4PL"):
        raise ValueError("Tanı yalnızca 3PL ve 4PL için tasarlanmıştır.")
    diagnostics=hessian_boundary_diagnostics(
        fit["items"],model,a_upper=fit.get("a_upper",4.),
        d_lower=fit.get("d_lower",.65))
    converged=bool(fit.get("converged",False))
    k=int(fit["k"])
    affected=int(diagnostics["Madde"].nunique())
    events=len(diagnostics)
    warnings=[]
    if not converged:
        warnings.append("EM yakınsama koşulu sağlanmadı.")
    if events:
        warnings.append(f"{events} parametre sınırda/yakın; {affected}/{k} madde etkilendi.")
    if model=="4PL":
        warnings.append("c ve d parametrelerinin birlikte kestirimi zayıf tanımlanabilirlik gösterebilir.")
    if not converged:
        decision="Güvenilir kestirim sağlanamadı"
    elif events:
        decision="İhtiyatla yorumlanmalı"
    else:
        decision="Tanılar açısından ön koşullar sağlandı; bağımsız doğrulama gerekli"
    return {
        "Model":model,"Karar":decision,"Yakınsama":converged,
        "Sınırdaki madde":affected,"Sınırdaki parametre":events,
        "Madde oranı":affected/k,"Açıklamalar":warnings,
        "Sınırlılık":"Bu karar parametre sınırları ve EM yakınsamasına dayanır. "
                     "Bilgi matrisi, model uyumu, yerel bağımsızlık, "
                     "karşılaştırmalı doğrulama ayrıca değerlendirilmelidir."
    }
