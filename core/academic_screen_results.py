"""Read-only screen-family adapters reusing the existing academic envelope.

The original pilot schema stays byte-identical for archived scientific evidence.
New domains delegate common-field validation, then retain their actual domain.
"""
import numpy as np
import pandas as pd
from core.academic_result_adapters import _clean
from core.academic_result_schema import build_result,validate_result
from core import reproducible as R

DOMAINS={'CTT','IRT','GT','DIF','MIRT','FACTOR','DESCRIPTIVE','DATA'}
AUXILIARY={
 '1. Ana Sayfa':'Tanıtım; hesaplanan analiz sonucu yok.',
 '2. Veri Yükleme':'Yardımcı dosya yükleme/eşleme ekranı; bağımsız bilimsel analiz değildir.',
 '3. Puanlama & Veri Hazırlama':'Yardımcı puanlama/ön işleme; işlem kodu korunur, bağımsız akademik sonuç raporu değildir.',
 '10. Raporlama':'Mevcut raporları birleştirir; yeni bağımsız analiz üretmez.',
 '11. Formüller ve Yöntemler':'Başvuru/yöntem metni; veri üzerinde analiz üretmez.',
 '12. Kullanım Kılavuzu':'Kullanım yardımı; veri üzerinde analiz üretmez.',
}
POSTPROCESS=('core.irt_python_mirt_report','core.irt_library','core.irt_binary_inference',
 'core.irt_python_mirt_parameters','core.irt_person_abilities','core.irt_b14_persons_info',
 'core.irt_b14_model_curves','core.irt_b14_diagnostics','core.academic_',
 'core.reproducible','reports.')


def domain(page):
    if page in AUXILIARY:return None
    if 'Kayıp' in page or 'Eksik Veri' in page:return 'DATA'
    if 'Genellenebilirlik' in page:return 'GT'
    if 'DIF' in page:return 'DIF'
    if 'MIRT' in page or 'Bifaktör' in page:return 'MIRT'
    if 'Boyutluluk' in page or 'Yerel Bağımsızlık' in page:return 'FACTOR'
    if 'MTK:' in page:return 'IRT'
    if 'Betimsel' in page or 'Grafikler' in page:return 'DESCRIPTIVE'
    return 'CTT'


def primary(call):
    return (call['function'] not in R.PREFLIGHT_ONLY and
            (call['module'],call['function'])!=('core.ctt_dual_engine','item_guidance') and
            not call['module'].startswith(POSTPROCESS))


def comparison_inputs_match(recorder,source):
    """A first model-comparison visit may import fits, but only of exact current data."""
    if not isinstance(source,pd.DataFrame):return False
    outputs=roots(recorder);seen=False
    for call in recorder.calls:
        if call['function']!='fit_python_mirt':continue
        args=R.decode(call['arguments'],[outputs.get(i) for i in range(len(recorder.calls))])
        x=args.get('responses',args.get('data',args.get('x')))
        if x is None or args.get('estimation','ML')!='ML':return False
        try:equal=np.array_equal(np.asarray(x),source.to_numpy(),equal_nan=True)
        except TypeError:return False
        if not equal:return False
        seen=True
    return seen


def roots(recorder):
    result={}
    for value in recorder.keep:
        entry=recorder.references.get(id(value));path=entry.get('path') if isinstance(entry,dict) else entry
        if path is not None and len(path)==1:result[path[0]]=value
    return result


def convergence(recorder):
    statuses=[]
    def visit(value):
        if isinstance(value,dict):
            for key in ('converged','Yakınsama'):
                if isinstance(value.get(key),(bool,np.bool_)):statuses.append(bool(value[key]))
            for key in ('fit','result','outcome','diagnostics'):
                if key in value:visit(value[key])
        elif isinstance(value,(list,tuple)):
            for child in value:visit(child)
        meta=getattr(value,'metadata',None)
        if isinstance(meta,dict) and isinstance(meta.get('Yakınsama'),(bool,np.bool_)):statuses.append(bool(meta['Yakınsama']))
        fit=getattr(value,'fit',value)
        flag=getattr(fit,'converged',None)
        if isinstance(flag,(bool,np.bool_)):statuses.append(bool(flag))
    for value in roots(recorder).values():visit(value)
    return None if not statuses else any(statuses)


def input_sample(recorder,fallback):
    values=roots(recorder);results=[values.get(i) for i in range(len(recorder.calls))]
    for call in recorder.calls:
        if not primary(call):continue
        try:args=R.decode(call['arguments'],results)
        except (IndexError,KeyError,TypeError,AttributeError):continue
        for key in ('data','x','responses','df','frame','X'):
            x=args.get(key)
            if isinstance(x,(pd.DataFrame,np.ndarray)) and x.ndim==2 and len(x):
                items=next((args[key] for key in ('items','item_columns','item_cols','columns') if isinstance(args.get(key),(list,tuple))),None)
                return int(len(x)),int(len(items) if isinstance(items,(list,tuple)) else x.shape[1]),'Kaydedilen hesaplama API girdisi'
    if isinstance(fallback,pd.DataFrame) and len(fallback):return len(fallback),len(fallback.columns),'Etkin veri girdisi; yöntem içindeki dışlamalar ilgili tablolarda'
    raise ValueError('Güvenilir örneklem bilgisi bulunamadı; N uydurulmadı.')


def validate_screen_result(result):
    family=result.get('family')
    if family not in DOMAINS:raise ValueError('Unknown academic screen domain')
    # Common invariant validation is identical for every family. No statistical
    # meaning is assigned to this temporary structural-validation projection.
    checked=validate_result({**result,'family':family if family in ('CTT','IRT','GT') else 'CTT'})
    checked['family']=family
    return checked


def adapt_screen(page,recorder,tables,metrics,notices,source,*,figure_count=0):
    family=domain(page)
    if family is None:raise ValueError(AUXILIARY[page])
    if convergence(recorder) is False:raise ValueError('Yakınsama sağlanmadı; geçerli akademik model raporu oluşturulamaz.')
    n,k,basis=input_sample(recorder,source)
    observed={}
    for title,frame in tables:
        if not isinstance(frame,pd.DataFrame) or frame.empty:continue
        if not isinstance(frame.index,pd.RangeIndex):frame=frame.reset_index()
        name=title or 'Hesaplanan tablo'
        if name in observed:name+=f' ({len(observed)+1})'
        observed[name]=_clean(frame)
    if not observed and not metrics and not figure_count:raise ValueError('Ekranda raporlanabilir hesaplanmış tablo/istatistik yok.')
    parameters=[];standard_errors=[]
    for rows in observed.values():
        keys=set().union(*(r.keys() for r in rows)) if rows else set()
        if keys & {'a','b','c','d','Madde','Eşik','Adım','Yük','Variance'}:parameters.extend(rows)
        if any('SH' in str(c) or str(c).startswith('SE') for c in keys):standard_errors.extend(rows)
    manifest=recorder.manifest()
    settings=[]
    for call in recorder.calls:
        if not primary(call):continue
        try:args=R.decode(call['arguments'],[roots(recorder).get(i) for i in range(len(recorder.calls))])
        except (IndexError,KeyError,TypeError,AttributeError):continue
        settings.append({'API':call['module']+'.'+call['function'],
            'ayarlar':{name:_clean(value) for name,value in args.items() if value is None or isinstance(value,(str,bool,int,float))}})
    warnings=list(dict.fromkeys(notices))
    warnings+=['Ekranda hesaplanmayan veya bilimsel kapıyla bastırılan istatistikler üretilmemiştir.',
        'Kod üretimi ve Word oluşturulması bağımsız bilimsel sertifika değildir. Kaynak analizdeki sınırlılıklar geçerlidir.']
    findings=[f'{page} sonuçları aşağıdaki {len(observed)} hesaplanmış tabloda sunulmaktadır. '
              'Örneklem kapsamı ve dışlamalar yöntem notlarıyla birlikte değerlendirilmelidir.']
    for label,value in metrics.items():
        if isinstance(value,(str,int,float,bool)):findings.append(f'Ekranda raporlanan {label}: {value}.')
    result=build_result(family=family if family in ('CTT','IRT','GT') else 'CTT',model=page,n=int(n),k=int(k) if k else None,
        parameters=parameters,standard_errors=standard_errors,fit_statistics=_clean(metrics),tables=observed,warnings=warnings,
        provenance={'source_commit':manifest['source_commit'],'Python':manifest['python'],'packages':manifest['packages'],
                    'recording_status':manifest['status'],'scientific_certification':'NOT IMPLIED; kaynak analiz uyarılarına bakınız'},
        metadata={'analysis_type':'captured_screen','sample_basis':basis,'settings':settings,'findings':findings,'figure_count':figure_count,'schema_extension':'screen-domain-v1'})
    result['family']=family
    return validate_screen_result(result)
