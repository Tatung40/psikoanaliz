"""Single source for DIF-center tables, CSV, Excel, ZIP and comparison plots."""
import io
import json
import zipfile
import matplotlib.pyplot as plt
import pandas as pd
from reports.irt_b14_report import safe_frame, export_xlsx
from core.dif_center import support_matrix


def tables(result):
    frames={'DIF':result['table'], 'Support':support_matrix(),
            'Settings':pd.DataFrame([{'Setting':k,'Value':str(v)} for k,v in result['metadata'].items()])}
    frames.update(result.get('extra_tables',{}))
    if result.get('theta') is not None:frames['Theta_sensitivity']=result['theta']
    return frames


def figure(result):
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    for method,part in result['table'].groupby('Method',sort=False):
        ax.plot(part.Item,part.p_adj,marker='o',label=method)
    ax.axhline(result['metadata']['Alpha'],color='red',linestyle='--',label='α')
    ax.set(ylabel='Düzeltilmiş p (yöntem içi)',title='Yöntem sonuçları — farklı null hipotezleri; üstünlük testi değil')
    ax.legend();return fig


def downloads(result):
    frames=tables(result)
    narrative=('# DIF Analiz Merkezi\n\n'+result['metadata']['Limitations']+
        '\n\nMH/LR: çapa toplamı + incelenen madde. SIBTEST: yalnız çapa toplamı. '
        'Ham bileşen p değerleri tamamlayıcıdır; karar yalnız seçilen hipotezin yöntem içi düzeltilmiş p değerine dayanır. '
        'Hesaplanamayan hipotezler kayıtta NaN kalır; çoklu düzeltmede önceden seçilmiş aile büyüklüğü korunur. '
        'ETS ve ΔR² sınıfları kendi yöntem tanımlarındadır; yöntemler arasında eşdeğer etki ölçeği değildir. '
        'SIBTEST etki sınıfı bağımsız doğrulanmadığından atanmaz. Lord: bağımsız grup kalibrasyonları ve çapa A/B belirsizliğini taşıyan tam delta kovaryansı; Wald: eşzamanlı M1/M2 ortak kovaryansı. Yöntemlerin null/ölçek varsayımları aynı değildir. Genel DIF-FWER sertifikası yok.\n\n'
        'Theta duyarlılığı varsa: tek havuzlanmış sabit-a/2PL kalibrasyonu, GH61 EAP; '
        'çıkarılan yanıtlar yalnız puanlama sırasında eksik sayılır; yeniden kalibrasyon yok. '
        'Delta theta gerçek yetenek hatası, RMSD veya yanlılık düzeltmesi değildir.\n\n'+
        json.dumps(result['metadata'],ensure_ascii=False,indent=2,default=str))
    fig=figure(result);buf=io.BytesIO();fig.savefig(buf,format='png',dpi=150);plt.close(fig)
    picture=buf.getvalue();xlsx=export_xlsx(frames);archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,frame in frames.items():z.writestr(name+'.csv',safe_frame(frame).to_csv(index=False))
        z.writestr('report.md',narrative);z.writestr('report.xlsx',xlsx);z.writestr('comparison.png',picture)
    return dict(csv=safe_frame(frames['DIF']).to_csv(index=False).encode('utf-8-sig'),xlsx=xlsx,
                zip=archive.getvalue(),png=picture,report=narrative.encode('utf-8'))
