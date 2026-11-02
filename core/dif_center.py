"""DIF orchestration only: existing statsmodels/mirt engines, no new estimator."""
import importlib.metadata
import numpy as np
import pandas as pd
from core.dif_classical import (_prepare, mantel_haenszel, logistic_dif, adjust_pvalues,
                                ets_mh_class, jodoin_gierl)
from core.python_mirt_dif import fingerprint

BASE_METHODS = ('MH', 'LR', 'SIBTEST')
METHODS = BASE_METHODS+('Wald DIF','Lord χ²')
LIMITATIONS = ('NOT CERTIFIED / FWER NOT CERTIFIED. Yöntemler farklı koşullu sıfır hipotezlerini test eder. '
    'Çapalar önceden seçilmiş ve DIF’siz varsayılır; otomatik madde silinmez. Düzeltme her yöntem için '
    'seçilen tek hipotezin incelenen madde ailesinde yapılır; yöntemler arasında ortak hata garantisi yoktur. '
    'Impact ve çapa seçimi çıkarımı etkiler. Aynı yönde sonuç yöntem üstünlüğü veya yanlılık kanıtı değildir.')


def support_matrix():
    frame = pd.DataFrame([
        ('SIBTEST', 'IMPLEMENTED', 'İkili; gerçek mirt.sibtest, sabit çapalar', 'β, SH, z/χ², sd, p; genel FWER/etki sınıfı sertifikası yok'),
        ('IRT-LR', 'IMPLEMENTED / LIMITED', 'Mevcut formel model sayfaları; public çok-grup ayrı', 'G²/sd/parametre farkı/eğriler; public mirt p/q kapalı'),
        ('Lord χ²', 'IMPLEMENTED / NOT CERTIFIED', '1PL/2PL; ayrı public mirt kalibrasyonları', 'Oakes + tam çapa Jacobianı; 1PL A=1, 2PL mean/sigma'),
        ('Raju', 'IMPLEMENTED / DESCRIPTIVE', 'İkili lojistik; ortak alt/üst asimptot', 'Tam θ doğrusu işaretli/işaretsiz alan; test/p/etki sınıfı kapalı'),
        ('MH', 'INDEPENDENTLY REFERENCED', 'İkili; mevcut MH/RBG motoru', 'χ², sd=1, OR, ΔMH, ETS A/B/C; genelleme koşullu'),
        ('Lojistik LR', 'IMPLEMENTED', 'İkili; ordinal için mevcut ayrı ekran', 'Uniform/nonuniform/omnibus LR, sd, p, omnibus ΔR²'),
        ('Wald DIF', 'IMPLEMENTED / NOT CERTIFIED', '1PL M1 / 2PL M2; mevcut eşzamanlı MML', 'Tam ortak kontrast kovaryansı; Lord ile aynı yöntem değildir'),
    ], columns=['Method', 'Evidence', 'Design', 'Outputs / restriction'])
    frame['Implemented'] = True
    frame['Numerically_validated'] = frame.Method.isin(['SIBTEST','Raju','MH','Lojistik LR','IRT-LR','Lord χ²','Wald DIF'])
    frame['Independent_reference'] = frame.Method.eq('MH')
    frame['Certified'] = False
    frame['Validation_scope'] = ['Uncorrected original/crossing identities; corrected API exercised',
        'Existing model-specific evidence; no general DIF-FWER certificate',
        'Anchor Jacobian and covariance identities; independent R fixture check recorded separately', 'Adaptive quadrature identity; no area SE/p',
        'statsmodels matched-stratum/RBG reference', 'Nested likelihood identities; no general FWER control',
        'Full shared-anchor covariance from existing MML; independent R fixture check recorded separately']
    return frame


def observed(data, *, items, group, reference, focal, anchors, studied, methods=BASE_METHODS,
             correction='Holm', alpha=.05, lr_test='omnibus', sib_method='original', sib_correction=True, model='2PL'):
    import mirt
    items, anchors, studied, methods = map(list, (items, anchors, studied, methods))
    if (not methods or len(set(methods)) != len(methods) or not set(methods) <= set(METHODS)
            or lr_test not in ('uniform', 'nonuniform', 'omnibus') or not 0 < alpha < 1):
        raise ValueError('Geçersiz yöntem/hipotez/alpha')
    if len(set(items)) != len(items) or group in items:
        raise ValueError('Madde adları benzersiz ve grup sütunundan ayrı olmalı')
    if (len(set(anchors)) < 2 or len(set(anchors)) != len(anchors) or not set(anchors) <= set(items)
            or not studied or len(set(studied)) != len(studied) or not set(studied) <= set(items)-set(anchors)):
        raise ValueError('En az iki ayrı çapa ve çapalardan ayrı incelenen maddeler gerekli')
    x, gf, _ = _prepare(data[items], data[group], reference, focal, False)
    if np.any(np.ptp(x, axis=0) == 0):
        raise ValueError('Sabit madde kestirilemez')
    matching = x[:, [items.index(a) for a in anchors]].sum(1)
    rows = []; extra_tables={}; parametric={}
    for method in set(methods)&{'Wald DIF','Lord χ²'}:
        if model not in ('1PL','2PL'):raise ValueError('Lord/Wald yalnız 1PL/2PL')
        try:
            if method=='Wald DIF':
                from core.irt_dif import wald_dif
                outcome=wald_dif(data[items],data[group],reference,focal,model=model,anchors=anchors,studied=studied)
            else:
                from core.dif_lord import lord_dif
                outcome=lord_dif(data[items],data[group],reference,focal,model=model,anchors=anchors,studied=studied)
            parametric[method]=outcome['table'].set_index('Item')
            extra_tables[('Wald' if method=='Wald DIF' else 'Lord')+'_covariance']=outcome['covariance']
        except (ValueError,TypeError,np.linalg.LinAlgError) as exc:
            parametric[method]=pd.DataFrame([dict(Item=i,Method=method,Statistic=np.nan,df=1 if model=='1PL' else 2,p=np.nan,
                Status='UNESTIMABLE / NOT CERTIFIED',Reason=str(exc)) for i in studied]).set_index('Item')
    for method in methods:
        family = []
        for item in studied:
            j = items.index(item); y = x[:, j]
            base = dict(Item=item, Method=method, Status='IMPLEMENTED / NOT CERTIFIED', Reason='')
            if method in parametric:
                base.update(parametric[method].loc[item].to_dict(), Effect=np.nan,Effect_metric='Physical parameter contrast',
                            Direction='Δa/Δb = focal−reference on linked scale', Effect_class='Not applicable',Hypothesis='b' if model=='1PL' else 'a,b')
            elif method == 'SIBTEST':
                r = mirt.sibtest(x.astype(int), gf, suspect_items=[j],
                    matching_items=[items.index(a) for a in anchors], focal_group=1,
                    method=sib_method, correction=sib_correction)
                base.update(r, Statistic=r['chi2'], df=r['df'], p=r['p_value'], Effect=r['beta'],
                    Effect_metric='β (reference−focal)', Direction=('Odak lehine' if r['beta'] < 0 else
                    'Referans lehine' if r['beta'] > 0 else 'Yön yok'), Effect_class='UNVERIFIED',
                    Hypothesis=sib_method)
                if sib_method=='crossing':
                    base['Effect_metric']='β_cross = |β_low−β_high| (işaretsiz)'
                    base['Direction']='Kesişimli; yön eşleştirme puanına bağlı, β_cross işaretsizdir'
            elif method == 'MH':
                r = mantel_haenszel(y, gf, matching+y)
                base.update(r, Statistic=r['chi2'], df=1, p=r['p'], Effect=r['delta'], Effect_metric='ETS ΔMH',
                    Direction=('Odak lehine' if r['delta'] > 0 else 'Referans lehine' if r['delta'] < 0 else 'Yön yok'),
                    Effect_class=ets_mh_class(r['delta'],r['se_log_or'],r['p'],alpha), Hypothesis='uniform')
            else:
                r = logistic_dif(y, gf, matching+y)
                base.update(r, Statistic=r['lr_'+lr_test], df=2 if lr_test=='omnibus' else 1,
                    p=r['p_'+lr_test], Effect=r['delta_r2'], Effect_metric='Omnibus Nagelkerke ΔR²',
                    Direction='Grup katsayısı yalnız eşit-eğim M1 koşulunda yorumlanır',
                    Effect_class=jodoin_gierl(r['delta_r2'],r['p_omnibus'],alpha) if lr_test=='omnibus' else
                        'UNSUPPORTED — sınıf yalnız omnibus ΔR² için', Hypothesis=lr_test)
            base['Reason'] = base.get('reason', base.get('Reason',''))
            if not (np.isfinite(base['Statistic']) and base['Statistic'] >= 0 and base['df'] > 0
                    and np.isfinite(base['p']) and 0 <= base['p'] <= 1):
                base['p'] = np.nan; base['Status'] = 'UNESTIMABLE / NOT CERTIFIED'
                base['Reason'] = base['Reason'] or 'Geçerli istatistik/SH/katman yok'
                base['Effect_class'] = base['Direction'] = 'Hesaplanamadı'
                for key in tuple(base):
                    if key.startswith('p_'):base[key] = np.nan
            family.append(base)
        raw = np.array([r['p'] for r in family]); valid = np.isfinite(raw)
        # Retain the preselected family size even when a hypothesis is unestimable.
        adjusted = adjust_pvalues(np.where(valid,raw,1.), correction)
        adjusted[~valid] = np.nan
        for row, q in zip(family, adjusted):
            row['p_adj'] = q
            row['Decision'] = 'Hesaplanamadı' if not np.isfinite(q) else 'DIF işareti (koşullu)' if q < alpha else 'Reddedilmedi'
            if method=='LR' and lr_test=='omnibus':
                row['Effect_class']=jodoin_gierl(row['Effect'],q,alpha) if np.isfinite(q) else 'Hesaplanamadı'
            if method not in parametric and not np.isfinite(row['Effect']):
                row['Effect_class']=row['Direction']='Hesaplanamadı'
        rows.extend(family)
    meta = dict(Items=items, Anchors=anchors, Studied=studied, Methods=methods, Reference=repr(reference),
        Focal=repr(focal), Correction=correction, Alpha=alpha, LR_test=lr_test, SIBTEST_method=sib_method,
        SIBTEST_correction=sib_correction, N_reference=int((gf==0).sum()), N_focal=int(gf.sum()),
        Excluded_other_or_missing_group=int(len(data)-len(gf)),
        Parametric_model=model, Wald_design='Existing simultaneous M1/M2; reference N(0,1), focal mean/SD free; GH61',
        Lord_design='Separate ML N(0,1), Oakes, GH61/tol1e-8/max_iter3000; complete anchor-link delta covariance',
        mirt=importlib.metadata.version('mirt'), Data_SHA256=fingerprint(data,()),
        Matching='MH/LR: anchor total + studied item; SIBTEST: anchor total only',
        Limitations=LIMITATIONS, Scientific_status='NOT CERTIFIED')
    return {'table':pd.DataFrame(rows), 'metadata':meta, 'extra_tables':extra_tables}


def raju_area(reference, focal):
    """Exact logistic area on the whole real line; NO asymptotic test/SE/p claim.

    Inputs are physical [a,b,c,u] on the SAME linked theta scale. Unequal
    asymptotes imply a divergent area, not a finite clipped-grid surrogate.
    """
    r, f = np.asarray(reference,float), np.asarray(focal,float)
    if r.shape != (4,) or f.shape != (4,) or not np.isfinite([r,f]).all():
        raise ValueError('Sonlu a,b,c,u gerekli')
    if min(r[0],f[0]) <= 0 or not (0 <= r[2] < r[3] <= 1 and 0 <= f[2] < f[3] <= 1):
        raise ValueError('Geçersiz lojistik parametre')
    if not np.array_equal(r[2:],f[2:]):
        raise ValueError('Farklı asimptotlarda tüm θ doğrusu alanı ıraksar')
    height = r[3]-r[2]; signed = height*(r[1]-f[1])
    if r[0] == f[0]:
        unsigned = abs(signed)
    else:
        t = f[0]*r[0]*(f[1]-r[1])/(f[0]-r[0])
        left = height*np.logaddexp(0,t)*(r[0]-f[0])/(f[0]*r[0])
        unsigned = abs(left)+abs(signed-left)
    if not np.isfinite([signed,unsigned]).all():raise ValueError('Alan sayısal olarak hesaplanamadı')
    return dict(Signed_area=float(signed),Unsigned_area=float(unsigned),p=np.nan,
                Status='DESCRIPTIVE / NOT CERTIFIED',Direction='Alan odak−referans')


def theta_sensitivity(data, *, items, removed, model='2PL', identifiers=None):
    """ONE pooled library calibration, EAP before/after masking items; NO refit."""
    import mirt
    from core.irt_python_mirt_adapter import fit_python_mirt
    items=list(items); removed=list(removed)
    if (model not in ('Rasch','1PL','2PL') or not removed or len(set(removed)) != len(removed)
            or not set(removed) <= set(items) or len(items)-len(removed) < 3):
        raise ValueError('Rasch/sabit-a 1PL/2PL ve en az üç korunan madde gerekli')
    x=data[items].to_numpy(float)
    n=len(x);ids=np.arange(1,n+1) if identifiers is None else np.asarray(identifiers)
    if len(ids)!=n or pd.isna(ids).any() or pd.Index(ids).has_duplicates:
        raise ValueError('Benzersiz, eksiksiz kişi kimlikleri gerekli')
    outcome=fit_python_mirt(x,model=model,n_quadpts=61)
    if not outcome.fit.converged:raise ValueError('Yakınsamayan modelden θ karşılaştırılmaz')
    masked=x.copy();masked[:,[items.index(c) for c in removed]]=np.nan
    before=mirt.fscores(outcome.fit,x,method='EAP',n_quadpts=61)
    after=mirt.fscores(outcome.fit,masked,method='EAP',n_quadpts=61)
    frame=pd.DataFrame({'ID':ids,'Theta_before':np.asarray(before.theta).reshape(-1),
        'Theta_after':np.asarray(after.theta).reshape(-1),'Posterior_SD_before':np.asarray(before.standard_error).reshape(-1),
        'Posterior_SD_after':np.asarray(after.standard_error).reshape(-1)})
    if not np.isfinite(frame.iloc[:,1:].to_numpy()).all():raise ValueError('Sonlu olmayan kişi puanı')
    frame['Delta_theta']=frame.Theta_after-frame.Theta_before
    return frame
