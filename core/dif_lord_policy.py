"""1PL Lord presentation/inference policy; reference numerics stay immutable."""
import numpy as np

STATUS='DESCRIPTIVE / EQUAL LATENT SCALE ASSUMPTION NOT VERIFIED'
WARNING=('1PL Lord testi, grupların örtük yetenek ölçeklerinin eşitliği varsayımına duyarlıdır. '
         'Bu varsayım doğrulanmadığından p değeri raporlanmamaktadır. '
         'Çıkarımsal DIF değerlendirmesi için Wald DIF veya uygun 2PL Lord analizini kullanınız.')
DESCRIPTIVE_NOTE='χ² ve parametre farkları betimseldir; χ²’nin çıkarımsal geçerliliği doğrulanmamıştır.'


def free_scale_reference_eligible():
    """Check the recorded numerical criteria too, never a PASS label alone."""
    import json
    from core.dif_lord_rasch import REFERENCE,reference_valid
    if not reference_valid():return False
    try:
        evidence=json.loads(REFERENCE.read_text())
        if evidence['R_window']!='[-12,12]' or set(evidence['comparisons'])!={'61','101'}:return False
        for result in evidence['comparisons'].values():
            relative=result['Statistic']['max_relative']
            if not np.isfinite(relative) or not 0<=relative<=1e-3:return False
            for group in ('R','F'):
                g=result[group];diag=g['R_diagnostics'];score=g['R_solution_common_GH101_score']
                if (not np.isfinite(score) or not 0<=score<=1e-5 or not g['Python_converged']
                        or not all(diag[k] for k in ('converged','interior','covariance_valid'))):return False
        return True
    except (OSError,KeyError,ValueError,TypeError):return False


def protect_result(result):
    """Fail closed for all current/legacy user reports, without changing estimators.

    No checkbox, model-name certification, or cutoff on Wald sigma_F can verify
    the equal-latent-scale assumption. Raw archived reference outputs are not edited.
    """
    if result['metadata'].get('Parametric_model')!='1PL':return result
    table=result['table']
    mask=table['Method'].eq('Lord χ²')
    warning=WARNING;status=STATUS
    if 'Lord_identification' in table:
        from core.dif_lord_rasch import IDENTIFICATION
        free_scale=table['Lord_identification'].eq(IDENTIFICATION)
        if free_scale.any() and free_scale_reference_eligible():mask=mask & ~free_scale
        elif free_scale.any() and free_scale[mask].all():
            warning='Serbest σ Rasch Lord yolu için bağımsız R kanıtı, kaynak kimliği veya ortak ızgara sayısal geçerlilik kapısı doğrulanmadı; p/q kapalıdır. Wald DIF veya uygun 2PL Lord analizini kullanınız.'
            status='DESCRIPTIVE / FREE-SD RASCH REFERENCE NOT VERIFIED'
    if not mask.any():return result
    table=table.copy();metadata=result['metadata'].copy()
    for column in ('p','q','p_adj','p_value','p_value_adjusted','p_adjusted','q_value','q (BH)','p (BH-FDR)'):
        if column in table:table.loc[mask,column]=np.nan
    descriptive=mask & np.isfinite(table['Statistic'])
    table.loc[descriptive,'Status']=status
    if 'Decision' in table:table.loc[descriptive,'Decision']='ÇIKARIM KAPALI — yalnız betimsel'
    if 'Reason' not in table:table['Reason']=''
    table.loc[mask,'Reason']=[(str(r)+' ' if str(r) else '')+warning+' '+DESCRIPTIVE_NOTE
                             if warning not in str(r) else str(r)
                             for r in table.loc[mask,'Reason'].fillna('')]
    metadata['Lord_inference_status']=status
    metadata['Lord_warning']=warning
    if warning not in metadata.get('Limitations',''):
        metadata['Limitations']=metadata.get('Limitations','')+' '+warning+' '+DESCRIPTIVE_NOTE
    return {**result,'table':table,'metadata':metadata}
