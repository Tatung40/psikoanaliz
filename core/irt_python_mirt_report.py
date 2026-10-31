"""Read-only, structured nine-model reports from verified Python mirt1.2.0 APIs."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from core.irt_python_mirt_parameters import python_mirt_binary_parameters
from core.irt_library_parameter_mapping import curves_from_normalized_table
from core.irt_library_diagnostics import library_fit_diagnostics
from core.irt_b14_model_curves import fitted_category_curves
from core.irt_b14_persons_info import information_table
from core.irt_binary_usage_policy import usage_status

MODELS=("Rasch","1PL","2PL","3PL","4PL","GRM","PCM","GPCM","NRM")
BINARY=MODELS[:5]

@dataclass
class LibraryReport:
    model: str
    metadata: dict
    parameters: pd.DataFrame=field(default_factory=pd.DataFrame)
    curves: pd.DataFrame=field(default_factory=pd.DataFrame)
    information: pd.DataFrame=field(default_factory=pd.DataFrame)
    warnings: list=field(default_factory=list)
    errors: list=field(default_factory=list)


def binary_evidence_label(outcome, *, common_discrimination=False, converged=None):
    """Presentation only: never rewrite adapter status codes or infer normality."""
    if outcome.model=='1PL' and common_discrimination:
        return 'NOT VERIFIED — ortak serbest a; sabit-a kanıtı aktarılamaz'
    if converged is not True:
        return 'Yakınsama doğrulanmadı — geçici sonuç; koşullu doğrulama uygulanmaz'
    if (outcome.estimation=='ML' and outcome.n_persons==1000 and outcome.n_items==12
            and outcome.solver_settings==dict(max_iter=500,tol=1e-4,n_quadpts=21)):
        return 'KOŞULLU DOĞRULANDI — referans çalışma: ML, N=1000, k=12, θ~N(0,1) varsayımı; bu veriye sertifika değil'
    return 'Koşullu doğrulama kanıtı mevcut; referans kapsamı ML, N=1000, k=12, θ~N(0,1); bu tasarıma genellenmez'


def _array(parameters,key,shape):
    if key not in parameters:raise ValueError(f"Python mirt model.parameters: {key} eksik")
    a=np.asarray(parameters[key],float)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError(f"Geçersiz {key} boyutu/değeri")
    return a


def parameter_table(fit,model,items,labels):
    if len(items)!=len(set(items)):raise ValueError("Madde adları benzersiz olmalı")
    if model in BINARY:
        table=python_mirt_binary_parameters(fit,model="1PL" if model=="Rasch" else model,item_names=items)
        a=table.a.to_numpy()
        if model=="Rasch" and not np.array_equal(a,np.ones(len(a))):raise ValueError("Rasch a=1 kısıtı sağlanmıyor")
        if model=="1PL" and not np.array_equal(a,np.full(len(a),a[0])):raise ValueError("1PL ortak a kısıtı sağlanmıyor")
        return table
    parameters=fit.model.parameters  # Public model API, NOT nonexistent FitResult.parameters.
    k,C=len(items),len(labels)
    if model=="NRM":
        a=_array(parameters,'slopes',(k,C));d=_array(parameters,'intercepts',(k,C))
        if not np.array_equal(a[:,0],np.zeros(k)) or not np.array_equal(d[:,0],np.zeros(k)):
            raise ValueError("NRM kategori0 referans kısıtı sağlanmıyor")
        return pd.DataFrame([{"Madde":item,"Kategori":str(labels[c]),"Kategori kodu":c,
            "Eğim (A)":a[i,c],"Kesişim (D)":d[i,c],"Referans":c==0} for i,item in enumerate(items) for c in range(C)])
    a=_array(parameters,'discrimination',(k,))
    if model=="PCM" and not np.array_equal(a,np.ones(k)):raise ValueError("PCM a=1 kısıtı sağlanmıyor")
    b=_array(parameters,'thresholds' if model=='GRM' else 'steps',(k,C-1))
    if model=='GRM' and (np.any(a<=0) or np.any(np.diff(b,axis=1)<=0)):raise ValueError("GRM sıralı eşikleri geçersiz")
    if model=='GPCM' and np.any(a<=0):raise ValueError("GPCM a pozitif olmalı")
    table=pd.DataFrame({"Madde":items,"a":a})
    for j in range(C-1):table[f'{"Eşik" if model=="GRM" else "Adım"} b{j+1}']=b[:,j]
    return table


def build_report(outcome, *, model, items, labels=(), theta=None):
    if model not in MODELS:raise ValueError("Desteklenmeyen rapor modeli")
    items=list(map(str,items));theta=np.linspace(-4,4,161) if theta is None else np.asarray(theta,float)
    if outcome.n_items!=len(items):raise ValueError("Madde sayısı sonuç nesnesiyle uyuşmuyor")
    fit=outcome.fit
    report=LibraryReport(model,dict(Model=model,N=outcome.n_persons,Madde=outcome.n_items,
        Motor=outcome.engine,Kestirim=outcome.estimation,Yakınsama=None))
    settings=getattr(outcome,'solver_settings',{})
    for name in ('max_iter','tol','n_quadpts'):
        if name in settings:report.metadata[name]=settings[name]
    try:
        diag=library_fit_diagnostics(fit)
        report.metadata.update(Yakınsama=diag['converged'],İterasyon=diag['iterations'],
            LogLik=diag['log_likelihood'],AIC=diag['aic'],BIC=diag['bic'],Serbest_parametre=diag['n_parameters'])
        if not diag['converged']:report.warnings.append("Yakınsama sağlanmadı; sayılar geçicidir, başarılı bilimsel sonuç değildir.")
    except (ValueError,TypeError,AttributeError) as e:report.errors.append(f"Kestirim tanı API'si: {e}")
    report.warnings.append("Bağımsız bilimsel doğrulama tamamlandı iddiası yok; bu rapor formel DIF veya parametre GA üretmez.")
    if model=='Rasch':report.warnings.append("Python Rasch: kütüphanenin 1PL/Rasch ailesi, a=1 ve θ~N(0,1). Yerel serbest-σ Rasch ile aynı ölçek değildir.")
    if model=='1PL':
        from mirt.models.dichotomous import OneParameterLogistic
        fixed_a = isinstance(fit.model, OneParameterLogistic)
        report.warnings.append("1PL: a=1 sabit, θ~N(0,1); kanıt yalnız incelenen koşullar içindir." if fixed_a else
            "Bu sekmede 1PL: ortak a serbest (2PL + eşitlik kısıtı), θ~N(0,1). Kütüphanenin varsayılan 1PL'si a=1'dir; önceki sabit-a doğrulaması buraya aktarılamaz.")
    if model=='NRM':report.warnings.append("Kategori0 referanstır; seçenekler ve θ yönü başarı sırası değildir.")
    policy,message=usage_status(model,outcome.estimation,outcome.n_persons,outcome.n_items)
    report.metadata['Kullanım durumu']=policy if policy!='not_applicable' else 'EXPERIMENTAL'
    if model in ('1PL','2PL'):
        report.metadata['Kullanım durumu']=binary_evidence_label(
            outcome, common_discrimination=model=='1PL' and not fixed_a,
            converged=report.metadata['Yakınsama'])
    if message:report.warnings.append(message)
    if outcome.estimation=='MAP':report.warnings.append('Motorun logLik/AIC/BIC değerleri MAP noktasındadır; standart ML maksimumuna dayalı model seçimi olarak yorumlanmamalıdır.')
    try:
        report.parameters=parameter_table(fit,model,items,labels)
        report.parameters['Yakınsama']=report.metadata['Yakınsama']
        report.parameters['Kestirim']=outcome.estimation
        report.parameters['Kullanım durumu']=report.metadata['Kullanım durumu']
    except (ValueError,TypeError,AttributeError,KeyError) as e:report.errors.append(f"Madde parametre API'si: {e}")
    try:
        if theta.ndim!=1 or len(theta)<5 or not np.isfinite(theta).all():raise ValueError("Theta ızgarası geçersiz")
        if model in BINARY:
            if report.parameters.empty:raise ValueError("Doğrulanmış ikili katsayı tablosu yok")
            native=report.parameters[['Madde','a','b','c','d']]
            c=curves_from_normalized_table(native,theta)
            report.curves=pd.DataFrame([dict(Madde=item,Theta=t,Kategori=str(cat),Olasılık=float(p if cat else 1-p))
                for i,item in enumerate(items) for q,t in enumerate(theta) for cat in (0,1) for p in [c['probability'][q,i]]])
            info=c['information'];tif=c['test_information']
            report.information=pd.DataFrame(info,columns=[f'I_{item}' for item in items])
            report.information.insert(0,'θ',theta);report.information['Test bilgisi']=tif
            report.information['SEM(θ)']=np.divide(1,np.sqrt(tif),out=np.full_like(tif,np.nan),where=tif>0)
        else:
            result=dict(result=fit,k=len(items),labels=tuple(labels),items=tuple(items))
            report.curves=fitted_category_curves(result,theta)
            report.information=information_table(result,theta)
    except (ValueError,TypeError,AttributeError,KeyError) as e:report.errors.append(f"Eğri/bilgi API'si: {e}")
    return report
