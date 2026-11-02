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
    if not mask.any():return result
    table=table.copy();metadata=result['metadata'].copy()
    for column in ('p','q','p_adj','p_value','p_value_adjusted','p_adjusted','q_value','q (BH)','p (BH-FDR)'):
        if column in table:table.loc[mask,column]=np.nan
    table.loc[mask,'Status']=STATUS
    if 'Decision' in table:table.loc[mask,'Decision']='ÇIKARIM KAPALI — yalnız betimsel'
    if 'Reason' not in table:table['Reason']=''
    table.loc[mask,'Reason']=[(str(r)+' ' if str(r) else '')+WARNING+' '+DESCRIPTIVE_NOTE
                             if WARNING not in str(r) else str(r)
                             for r in table.loc[mask,'Reason'].fillna('')]
    metadata['Lord_inference_status']=STATUS
    metadata['Lord_warning']=WARNING
    if WARNING not in metadata.get('Limitations',''):
        metadata['Limitations']=metadata.get('Limitations','')+' '+WARNING+' '+DESCRIPTIVE_NOTE
    return {**result,'table':table,'metadata':metadata}
