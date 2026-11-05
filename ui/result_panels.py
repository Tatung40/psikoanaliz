"""Presentation helpers only: no estimation, state mutation or report generation."""
import inspect
import re
import unicodedata
import streamlit as st

OUTPUTS={'dataframe','table','pyplot','plotly_chart','json','line_chart','bar_chart',
         'area_chart','scatter_chart','altair_chart','vega_lite_chart','image'}


def report_name(page,state):
    if page=='Çok Boyutlu MTK (MIRT)' and isinstance(state.get('md_model'),str):
        return state['md_model']+' MIRT'
    match=re.fullmatch(r'MTK: (?:Python mirt )?(Rasch|1PL|2PL|3PL|4PL|GRM|PCM|GPCM|RSM|NRM)',page)
    if match:return match.group(1)
    dynamic={'MTK: Çok Kategorili B14':('b14_model',''),
             'MTK: Çok Kategorili Formel DIF':('poly_model',' DIF'),
             'MTK: Python mirt DIF':('pmd_model',' DIF'),
             'MTK: 3PL/4PL Bootstrap DIF':('b34_model',' Bootstrap DIF')}
    if page in dynamic:
        key,suffix=dynamic[page];model=state.get(key)
        if model in ('Rasch','1PL','2PL','3PL','4PL','GRM','PCM','GPCM','RSM','NRM'):return model+suffix
    names={'KTK: Çift Motor Rapor':'KTK Çift Motor','10. Raporlama':'KTK Toplu Rapor',
           'MTK: Karşılaştırma':'MTK Model Karşılaştırması','MTK: GRM Formel DIF':'GRM DIF',
           'Çok Boyutlu MTK (MIRT)':'Çok Boyutlu MTK (MIRT)',
           'Genellenebilirlik Kuramı (G ve D Çalışmaları)':'Genellenebilirlik Kuramı',
           'Eksik Veri ve Atama':'Kayıp Veri ve Atama'}
    return names.get(page,re.sub(r'^(?:\d+[A-Z]?\.\s*|MTK:\s*)','',page).strip())


def docx_download(page,state):
    name=report_name(page,state)
    slug=unicodedata.normalize('NFKD',name.replace('ı','i')).encode('ascii','ignore').decode().lower()
    slug=re.sub(r'[^a-z0-9]+','_',slug).strip('_')
    return name+' Akademik Rapor',slug+'_akademik_rapor.docx'


def target_for(original,values):
    return values.get('self') or getattr(original,'__self__',None) or st._main


def in_expander(target):
    return 'expandable' in target._active_dg._ancestor_block_types


def call_in(original,args,kwargs,target):
    """Rebind DG methods so explicit column/container calls use the intended panel."""
    if inspect.ismethod(original):return original.__func__(target,*args,**kwargs)
    if 'self' in inspect.signature(original).parameters:
        if args:return original(target,*args[1:],**kwargs)
        return original(**{**kwargs,'self':target})
    with target:return original(*args,**kwargs)


def panel_title(kind,values,heading):
    title=''
    if kind=='pyplot':
        fig=values.get('fig')
        if fig is not None and fig.axes:title=fig.axes[0].get_title()
    elif kind=='plotly_chart':
        fig=values.get('figure_or_data');layout=getattr(fig,'layout',None)
        title=getattr(getattr(layout,'title',None),'text',None) or ''
    elif kind in ('dataframe','table'):
        data=values.get('data');data=getattr(data,'data',data) if type(data).__name__=='Styler' else data
        columns={str(c) for c in getattr(data,'columns',())}
        if columns & {'Birey','Yetenek (θ)','theta','EAP','Posterior SD'}:title='Birey Parametreleri'
        elif any('DIF' in c for c in columns):title='DIF Sonuçları'
        elif columns & {'LogLik','loglik','AIC','BIC'}:title='Model Uyum İstatistikleri'
        elif 'Madde' in columns and (columns & {'a','b','c','d','Eğim (A)'}):title='Madde Parametreleri'
    return re.sub(r'<[^>]+>','',str(title or heading or ('Sonuç Tablosu' if kind in ('dataframe','table') else 'Tanı Grafikleri'))).strip('# ')


def render_output(ctx,kind,values,original,args,kwargs):
    target=target_for(original,values)
    if in_expander(target):return original(*args,**kwargs)
    title=panel_title(kind,values,ctx['heading'])
    count=ctx['panels'].get(title,0)+1;ctx['panels'][title]=count
    if count>1:title+=f' ({count})'
    panel=target.expander(title,expanded=False)
    return call_in(original,args,kwargs,panel)
