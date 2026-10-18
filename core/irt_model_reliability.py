"""Conservative, descriptive 3PL/4PL calibration reliability assessment.

Not a statistical goodness-of-fit test or evidence of construct validity.
Never labels a fit scientifically validated solely because EM converged.
"""
from core.irt import structural_identifiability
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
    structure=structural_identifiability(k,model)
    convergence=fit.get("convergence") or {}
    refinement=fit.get("refinement") or {}
    refined=bool(refinement.get("requested"))
    method="EM + L-BFGS-B" if refined else "EM"
    params_ok=convergence.get("parameter_criterion_met",True)
    affected=int(diagnostics["Madde"].nunique())
    events=len(diagnostics)
    warnings=[]
    if not structure["satisfied"]:
        warnings.append(structure["reason"])
    else:
        warnings.append(f"Yapısal sayım: {structure['parameters']} serbest parametre, en fazla "
                        f"{structure['pattern_df']} yanıt örüntüsü serbestlik derecesi. Bu koşul "
                        "gereklidir ama tanımlanabilirliği kanıtlamaz.")
    if refined:
        em_ok=refinement.get("em_converged",convergence.get("loglik_converged"))
        stage=("EM aşaması logL ölçütüyle yakınsadı" if em_ok
               else "EM aşaması logL ölçütüyle yakınsamadı")
        if not refinement.get("accepted"):
            warnings.append(f"Kestirim yöntemi: {method}. İyileştirme reddedildi "
                            f"({refinement.get('reason','neden belirtilmedi')}); EM sonucu raporlanıyor. {stage}.")
        else:
            warnings.append(f"Kestirim yöntemi: {method}. {stage}; L-BFGS-B iyileştirmesi kabul edildi, "
                            f"KKT ihlali {refinement.get('max_kkt_final',float('nan')):.2g}/kişi. "
                            "KKT koşulu yerel birinci mertebe koşuludur; global optimumu, "
                            "tanımlanabilirliği veya model geçerliğini kanıtlamaz.")
    if not converged:
        warnings.append("Yakınsama koşulu sağlanmadı." if refined else "EM yakınsama koşulu sağlanmadı.")
    elif not params_ok:
        warnings.append(convergence.get("message","Parametre değişimi ölçütü sağlanmadı."))
    if events:
        warnings.append(f"{events} parametre sınırda/yakın; {affected}/{k} madde etkilendi. "
                        "Bu parametreler için Hessian SH ve Wald aralıkları hesaplanmaz.")
    if model=="4PL":
        warnings.append("c ve d parametrelerinin birlikte kestirimi zayıf tanımlanabilirlik gösterebilir.")
    if not structure["satisfied"]:
        decision="Yapısal olarak tanımlanamaz"
    elif not converged:
        decision="Güvenilir kestirim sağlanamadı"
    elif events or not params_ok:
        decision="İhtiyatla yorumlanmalı"
    else:
        decision="Tanılar açısından ön koşullar sağlandı; bağımsız doğrulama gerekli"
    return {
        "Model":model,"Karar":decision,"Yakınsama":converged,"Kestirim yöntemi":method,
        "Yakınsama ölçütü":convergence.get("criterion","loglik"),
        "Parametre yakınsaması":bool(params_ok),
        "Serbest parametre":structure["parameters"],
        "Örüntü serbestlik derecesi":structure["pattern_df"],
        "Sınırdaki madde":affected,"Sınırdaki parametre":events,
        "Madde oranı":affected/k,"Açıklamalar":warnings,
        "Sınırlılık":"Bu karar yapısal parametre sayımı, sınır kestirimleri ve yakınsama "
                     "tanısına (EM logL ölçütü ya da iyileştirme sonrası KKT ölçütü) dayanır. Pozitif tanımlı bir bilgi matrisi ya da madde "
                     "önselleri tanımlanabilirliği kendiliğinden sağlamaz. Bilgi matrisi, "
                     "model uyumu, yerel bağımsızlık ve karşılaştırmalı doğrulama ayrıca "
                     "değerlendirilmelidir."
    }
