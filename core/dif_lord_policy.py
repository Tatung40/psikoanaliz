"""1PL Lord presentation/inference policy; reference numerics stay immutable."""
import numpy as np

STATUS='DESCRIPTIVE / EQUAL LATENT SCALE ASSUMPTION NOT VERIFIED'
WARNING=('1PL Lord testi, grupların örtük yetenek ölçeklerinin eşitliği varsayımına duyarlıdır. '
         'Bu varsayım doğrulanmadığından p değeri raporlanmamaktadır. '
         'Çıkarımsal DIF değerlendirmesi için Wald DIF veya uygun 2PL Lord analizini kullanınız.')
DESCRIPTIVE_NOTE='χ² ve parametre farkları betimseldir; χ²’nin çıkarımsal geçerliliği doğrulanmamıştır.'


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
        from core.dif_lord_rasch import IDENTIFICATION,reference_valid
        free_scale=table['Lord_identification'].eq(IDENTIFICATION)
        if free_scale.any() and reference_valid():mask=mask & ~free_scale
        elif free_scale.any() and free_scale[mask].all():
            warning='Serbest σ Rasch Lord yolu için bağımsız R sayısal kanıtı yok veya kaynak kimliği değişmiş; p/q kapalıdır. Wald DIF veya uygun 2PL Lord analizini kullanınız.'
            status='DESCRIPTIVE / FREE-SD RASCH REFERENCE NOT VERIFIED'
    if not mask.any():return result
    table=table.copy();metadata=result['metadata'].copy()
    for column in ('p','q','p_adj','p_value','p_value_adjusted','p_adjusted','q_value','q (BH)','p (BH-FDR)'):
        if column in table:table.loc[mask,column]=np.nan
    table.loc[mask,'Status']=status
    if 'Decision' in table:table.loc[mask,'Decision']='ÇIKARIM KAPALI — yalnız betimsel'
    if 'Reason' not in table:table['Reason']=''
    table.loc[mask,'Reason']=[(str(r)+' ' if str(r) else '')+warning+' '+DESCRIPTIVE_NOTE
                             if warning not in str(r) else str(r)
                             for r in table.loc[mask,'Reason'].fillna('')]
    metadata['Lord_inference_status']=status
    metadata['Lord_warning']=warning
    if warning not in metadata.get('Limitations',''):
        metadata['Limitations']=metadata.get('Limitations','')+' '+warning+' '+DESCRIPTIVE_NOTE
    return {**result,'table':table,'metadata':metadata}
