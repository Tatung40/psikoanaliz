"""Native CFA postprocessing; no estimator or inferential reference is introduced."""
import pandas as pd
from core.irt_mirt_policy import numerical_scope
from core.irt_mirt_diagnostics import enrich
CORE_MODELS=frozenset(("2PL","GRM","GPCM","PCM","Rasch","1PL (a=1)","NRM"))
def _owned(outcome):
    return outcome.get("model") in CORE_MODELS and not outcome.get("exploratory",False)
def refresh_policy(outcome,result):
    """Evidence label refresh without fitting, scoring or old-snapshot mutation."""
    if not _owned(outcome):return result
    out={"metadata":dict(result["metadata"]),"tables":dict(result["tables"])}
    meta,tables=out["metadata"],out["tables"]
    meta.update(numerical_scope(outcome))
    note=("Bağımsız R kapsamı yöntem/ayar iddiasıdır; kullanıcı verisinin örtük üreteci doğrulanmış sayılmaz. "
          "Madde SH/GA/p ve biçimsel uyum/LR çıkarımı kapalıdır. EAP posterior SH/kovaryans sonlu GH ızgarasına ve "
          "kestirilmiş madde parametrelerine koşulludur; frequentist kapsama değildir. "
          "IRT ayırıcılık vektörleri standartlaştırılmış FA yükleri değildir; genel skaler çok boyutlu güçlük yoktur.")
    meta["Limitations"]=str(meta.get("Limitations",""))+" "+note
    meta["Production_status"]="SCOPED NUMERICAL PRODUCTION" if meta.get("Numerical_status","").startswith("NUMERICALLY CERTIFIED") else "EXPERIMENTAL / NOT CERTIFIED"
    model=outcome.get("model")
    if model=="NRM":
        parameterization="Native nominal category-specific slope vectors/intercepts, category0 zero reference; no ordered thresholds or universal multidimensional difficulty."
        parameter_evidence="Native nominal category coefficients; cold equivalence only within recorded C1 scope. Warm evidence alone is not cold certification."
    elif model in ("2PL","Rasch","1PL (a=1)"):
        parameterization="Native binary slope vector and direct intercept d; no universal multidimensional scalar difficulty."
        parameter_evidence="Direct native slopes/intercepts within the archived binary profile; no item SE/CI."
    else:
        parameterization="Native thresholds/steps b; transformed d=-sum(a)*b. Not universal multidimensional difficulty."
        parameter_evidence="GRM/GPCM/PCM transformed category intercept mapping; not raw-threshold equivalence."
    meta["Threshold_parameterization"]=parameterization
    definitions={
      "2PL":"P(X=1|theta)=logistic(a' theta+d); D=1.",
      "Rasch":"P(X=1|theta)=logistic(sum(active theta)+d); active loadings fixed1, structural zeros fixed0; zero means/unit latent variances, not a free-SD Rasch model.",
      "1PL (a=1)":"Fixed-unit-vector Rasch-type alias; no free common discrimination.",
      "GRM":"P(X>=k|theta)=logistic(a' theta-sum(a)*b_k); ordered native thresholds; category probabilities are adjacent differences.",
      "GPCM":"P(X=k|theta) proportional exp(k*a' theta-sum(a)*sum(native steps_1..k)); reference category0.",
      "PCM":"Same native GPCM category equation, with every declared active loading fixed1; step intercepts remain item-specific.",
      "NRM":"P(X=k|theta) proportional exp(a_k' theta+d_k); category0 a,d fixed0; signed category-specific vectors, no ordinal scoring."
    }
    meta["Model_definition"]=definitions.get(model,"See engine specification")
    meta["Model_comparisons"]="ML logLik/AIC/BIC descriptive only; identical data/categories/likelihood/grid; no Wilks LRT"
    status=meta.get("Status","EXPERIMENTAL / NOT CERTIFIED")
    conditions=str(meta.get("Applicable_conditions","Exact archived profile only"))
    rows=[]
    for output,qualification in (
        ("Item parameters in native model-specific coordinates",parameter_evidence),
        ("Latent covariance/correlation","Zero means/unit variances; declared free correlations only"),
        ("Category probabilities","R comparison on [-2,0,2]^q; other sections are calculation extrapolations"),
        ("Item information matrices","Independent R matrices on registered grid; no universal scalar"),
        ("Test/directional information","Native item-matrix sum; explicit direction e' I e"),
        ("EAP/posterior SD","R comparison first16 fixture persons; conditional posterior, not frequentist coverage"),
        ("logLik/AIC/BIC/count","Matched finite-grid ML; descriptive summaries only")):
        rows.append(dict(Output=output,Numerical_status=status,Inferential_status="WITHHELD",Conditions=conditions,Qualification=qualification))
    rows.append(dict(Output="MAP person mode",Numerical_status="EXPERIMENTAL / conditional calculation only",
                     Inferential_status="WITHHELD",Conditions="Native fscores posterior mode",
                     Qualification="No independent R MAP-person comparison or frequentist uncertainty certificate"))
    rows.append(dict(Output="Full posterior covariance",Numerical_status=meta.get("Posterior_covariance_status","WITHHELD"),
                     Inferential_status="WITHHELD / no coverage certification",Conditions="Conditional finite-grid public posterior",
                     Qualification="Weighted centered-node identity; not independently R-certified covariance"))
    rows.extend(dict(Output=name,Numerical_status="WITHHELD",Inferential_status="WITHHELD",Conditions="No validated inferential method",
                     Qualification="Not supplied as p/SE/CI") for name in ("Item SE/CI","Global/item-fit p","Wilks comparison"))
    tables["Output_scope"]=pd.DataFrame(rows)
    tables["Summary"]=pd.DataFrame([dict(Setting=str(k),Value=str(v)) for k,v in meta.items()])
    return out
def decorate_native(outcome,result,person_ids=None,extra_metadata=None):
    """Postprocess one native report. Other engines/structures remain unchanged."""
    if not _owned(outcome):return result
    out={"metadata":dict(result["metadata"]),"tables":dict(result["tables"])}
    from time import perf_counter
    if extra_metadata:out["metadata"].update(extra_metadata)
    started=perf_counter()
    out=enrich(outcome,out,person_ids=person_ids)
    out["metadata"]["Derived_diagnostics_seconds"]=perf_counter()-started
    return refresh_policy(outcome,out)
