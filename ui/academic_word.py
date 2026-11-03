"""Session-local, read-only academic export of the outputs actually shown by a page."""
from contextvars import ContextVar
from functools import wraps
import hashlib
import inspect
import io
import json
import re
import sys
import numpy as np
import pandas as pd
import streamlit as st
from core.academic_screen_results import adapt_screen,comparison_inputs_match,convergence,domain,primary
from core import reproducible as R

# Stable dispatcher survives the existing in-place-deploy module guard. User data
# is kept only in the current ContextVar/session, never in a shared global cache.
CTX=sys.__dict__.setdefault('_academic_screen_ctx',ContextVar('academic_screen_ctx',default=None))
BUSY=sys.__dict__.setdefault('_academic_screen_busy',ContextVar('academic_screen_busy',default=False))
WIDGETS={'selectbox','multiselect','radio','checkbox','toggle','slider','select_slider',
         'number_input','text_input','text_area','date_input','file_uploader','data_editor'}
DISPLAY={'dataframe','table','metric','pyplot','plotly_chart','json','markdown','caption',
         'header','subheader','title','warning','error','info','success','download_button'}
PRESENTATION=re.compile(r'(?:output_|chart_|show_|plot_|report_|theta_range|theta_min|theta_max|theta_points)')


def _digest(value):
    if isinstance(value,pd.DataFrame):
        return hashlib.sha256(pd.util.hash_pandas_object(value,index=True).to_numpy().tobytes()+
            repr((tuple(value.columns),tuple(map(str,value.dtypes)))).encode()).hexdigest()
    if hasattr(value,'getvalue'):return hashlib.sha256(value.getvalue()).hexdigest()
    if isinstance(value,(tuple,list)):return [_digest(v) for v in value]
    if hasattr(value,'item'):
        try:return value.item()
        except (ValueError,TypeError):pass
    if isinstance(value,(str,int,float,bool)) or value is None:return value
    return str(value)


def _frame(value,kwargs):
    if type(value).__name__=='Styler':value=value.data
    if isinstance(value,pd.Series):value=value.to_frame()
    if not isinstance(value,pd.DataFrame):
        try:value=pd.DataFrame(value)
        except (ValueError,TypeError):return None
    columns=list(value.columns)
    config=kwargs.get('column_config')
    if isinstance(config,dict):columns=[c for c in columns if c not in config or config[c] is not None]
    if kwargs.get('column_order') is not None:columns=[c for c in kwargs['column_order'] if c in columns]
    return value.loc[:,columns].copy(deep=True)


def _observe(name,args,kwargs,original):
    ctx=CTX.get()
    if ctx is None or BUSY.get():return original(*args,**kwargs)
    try:values=dict(inspect.signature(original).bind_partial(*args,**kwargs).arguments)
    except (TypeError,ValueError):return original(*args,**kwargs)
    values.update(values.pop('kwargs',{}))
    # Preserve existing specialized Word exporters and their location, but do not
    # enqueue their download until current-data/settings provenance is confirmed.
    if name=='download_button':
        filename=values.get('file_name','') or ''
        if str(filename).lower().endswith('.docx'):
            target=values.pop('self',None) or getattr(original,'__self__',None) or st
            slot=target.empty();ctx['existing'].append((slot,values));return False
        return original(*args,**kwargs)
    prepared=None
    if name=='pyplot':
        fig=values.get('fig')
        if fig is None:
            import matplotlib.pyplot as plt
            fig=plt.gcf()
        out=io.BytesIO();fig.savefig(out,format='png',dpi=130)
        prepared=(ctx['heading'] or 'Ekrandaki grafik',out.getvalue())
    token=BUSY.set(True)
    try:result=original(*args,**kwargs)
    finally:BUSY.reset(token)
    if name in WIDGETS:
        label=str(values.get('label',''));key=str(values.get('key') or name+':'+label)
        ctx['view_widgets'][key]=_digest(result)
        if not PRESENTATION.search(key):ctx['widgets'][key]=_digest(result)
        if name=='data_editor' and isinstance(result,pd.DataFrame):
            ctx['tables'].append((ctx['heading']+' — düzenlenen model yapısı',result.copy(deep=True)))
    elif name in ('dataframe','table'):
        frame=_frame(values.get('data'),values)
        if frame is not None:ctx['tables'].append((ctx['heading'],frame))
    elif name=='metric':ctx['metrics'][str(values.get('label','İstatistik'))]=_digest(values.get('value'))
    elif name=='pyplot':ctx['figures'].append(prepared)
    elif name=='plotly_chart':
        import plotly.graph_objects as go
        ctx['plotly'].append((ctx['heading'] or 'Ekrandaki grafik',go.Figure(values.get('figure_or_data'))))
    elif name in ('header','subheader','title','markdown'):
        text=str(values.get('body',''))
        if name!='markdown' or text.startswith('#'):ctx['heading']=text.lstrip('# ').strip()
        elif text and len(text)<1500:ctx['notes'].append(text)
    elif name in ('warning','error','info','success','caption'):
        text=str(values.get('body',''))
        if text and len(text)<4000:ctx['notes'].append(text)
    elif name=='json':
        value=values.get('body')
        if isinstance(value,dict):
            ctx['notes'].append('Ekran yöntem/tanı bilgisi: '+str({k:v for k,v in value.items() if isinstance(v,(str,int,float,bool)) or v is None}))
    return result


def _install():
    from streamlit.delta_generator import DeltaGenerator
    sys.__dict__['_academic_screen_dispatch']=_observe
    for obj in (st,DeltaGenerator):
        for name in WIDGETS|DISPLAY:
            original=getattr(obj,name,None)
            if original is None or getattr(original,'__academic_capture__',False):continue
            def wrap(fn,kind):
                @wraps(fn)
                def call(*a,**kw):return sys.__dict__['_academic_screen_dispatch'](kind,a,kw,fn)
                call.__academic_capture__=True
                return call
            setattr(obj,name,wrap(original,name))


def begin(page,raw=None,scored=None,items=()):
    _install()
    sources=[x for x in (raw,scored) if isinstance(x,pd.DataFrame)]
    source=scored if isinstance(scored,pd.DataFrame) else raw
    if isinstance(source,pd.DataFrame) and items:source=source[[c for c in items if c in source]]
    data_key=hashlib.sha256(json.dumps([_digest(x) for x in sources]).encode()).hexdigest()
    if st.session_state.get('_academic_data_key')!=data_key:
        st.session_state.pop('_academic_history',None)
        st.session_state['_academic_data_key']=data_key
    CTX.set(dict(page=page,source=source,data_key=data_key,widgets={},view_widgets={},tables=[],metrics={},
                 notes=[],figures=[],plotly=[],heading='',existing=[]))


def _plotly_png(caption,fig):
    """Render already-computed trace points; no new estimates or curve interpolation."""
    import matplotlib.pyplot as plt
    graph,ax=plt.subplots(figsize=(6.4,3.8));drawn=False
    if fig.data and all(t.type=='surface' for t in fig.data):
        ax.remove();ax=graph.add_subplot(111,projection='3d')
        for trace in fig.data:
            z=np.asarray(trace.z,float)
            x=np.asarray(trace.x if trace.x is not None else np.arange(z.shape[1]),float)
            y=np.asarray(trace.y if trace.y is not None else np.arange(z.shape[0]),float)
            if x.ndim==y.ndim==1:x,y=np.meshgrid(x,y)
            ax.plot_surface(x,y,z);drawn=True
        ax.set_zlabel(str(fig.layout.scene.zaxis.title.text or ''))
        out=io.BytesIO();graph.savefig(out,format='png',dpi=130);plt.close(graph)
        return caption,out.getvalue()
    if any(t.type not in ('scatter','scattergl','bar','heatmap','indicator') for t in fig.data):
        plt.close(graph);return None  # Never silently present a partial figure.
    for trace in fig.data:
        kind=trace.type;label=str(trace.name or '')
        if kind in ('scatter','scattergl','bar') and trace.y is not None:
            y=list(trace.y);x=list(trace.x) if trace.x is not None else list(range(len(y)))
            if len(x)!=len(y):continue
            if kind=='bar':ax.bar(x,y,label=label)
            else:ax.plot(x,y,label=label,linestyle='-' if 'lines' in (trace.mode or 'lines') else 'None',marker='o' if 'markers' in (trace.mode or '') else None)
            drawn=True
        elif kind=='heatmap' and trace.z is not None:
            z=np.asarray(trace.z,float)
            if z.ndim==2:ax.imshow(z,aspect='auto',interpolation='nearest');drawn=True
        elif kind=='indicator' and trace.value is not None:
            ax.text(.5,.5,str(trace.value),ha='center',va='center',fontsize=22);ax.set_axis_off();drawn=True
    if not drawn:plt.close(graph);return None
    ax.set_title(str(fig.layout.title.text or caption));ax.set_xlabel(str(fig.layout.xaxis.title.text or ''))
    ax.set_ylabel(str(fig.layout.yaxis.title.text or ''))
    handles,labels=ax.get_legend_handles_labels()
    if labels:ax.legend(fontsize=7)
    out=io.BytesIO();graph.tight_layout();graph.savefig(out,format='png',dpi=130);plt.close(graph)
    return caption,out.getvalue()


def finish(recorder):
    ctx=CTX.get();CTX.set(None)
    if ctx is None:return
    page=ctx['page']
    if domain(page) is None:
        for slot,values in ctx['existing']:slot.download_button(**values)
        return  # Scope reason is explicit in the coverage matrix.
    material=[(i,c) for i,c in enumerate(recorder.calls) if primary(c)]
    imported=set(recorder.imported.values())
    fresh=any(i not in imported for i,c in material)
    if page=='MTK: Karşılaştırma' and comparison_inputs_match(recorder,ctx['source']):fresh=True
    signature=hashlib.sha256(json.dumps([ctx['data_key'],ctx['widgets']],sort_keys=True,default=str).encode()).hexdigest()
    history=st.session_state.setdefault('_academic_history',{});old=history.get(page)
    reason=None
    if not material:reason='Geçerli üretici çağrısı kaydedilmedi; analizi bu oturumda çalıştırınız.'
    elif not fresh and (old is None or old['signature']!=signature):reason='Veri/model/ayar değişti veya eski oturum sonucu doğrulanamadı; eski Word ve Python kodu kapatıldı. Analizi yeniden çalıştırınız.'
    if convergence(recorder) is False:reason='Kestirim yakınsamadı; geçerli akademik rapor üretilmedi.'
    encoding_errors=[e for e in recorder.errors if 'no faithful public serialization' in e]
    if encoding_errors:reason='Gerekli üretici kaydı yok; opak model nesnesi uydurularak kod/rapor oluşturulmadı. Analizi yeniden çalıştırınız.'
    if reason:
        history.pop(page,None)
        if material and not any(reason in e for e in recorder.errors):recorder.errors.append(reason)
        for slot,values in ctx['existing']:slot.warning(reason)
        if material:st.caption('Akademik Word: '+reason)
        return
    history[page]={'signature':signature}
    if ctx['existing']:
        history[page]['legacy_docx']=[v.get('data') for _,v in ctx['existing']]
        for slot,values in ctx['existing']:slot.download_button(**values)
        return
    try:
        result=adapt_screen(page,recorder,ctx['tables'],ctx['metrics'],ctx['notes'],ctx['source'],figure_count=len(ctx['figures'])+len(ctx['plotly']))
        # Presentation choices do not demand a new calibration, but a Word
        # cache must still follow the currently displayed figures and inputs.
        visual=hashlib.sha256(json.dumps(ctx['view_widgets'],sort_keys=True,default=str).encode())
        for caption,png in ctx['figures']:visual.update(caption.encode());visual.update(png)
        for caption,fig in ctx['plotly']:visual.update(caption.encode());visual.update(fig.to_json().encode())
        result['provenance'].update(input_fingerprint=ctx['data_key'],view_fingerprint=visual.hexdigest())
        if not fresh and old and old.get('result')==result:
            payload=old['docx']
        else:
            from reports.academic_docx import build_academic_docx
            from core.academic_screen_results import validate_screen_result
            figures=list(ctx['figures'])
            for caption,fig in ctx['plotly']:
                image=_plotly_png(caption,fig)
                if image is not None:figures.append(image)
                else:result['warnings'].append(caption+': etkileşimli grafik statik biçime çevrilemedi; sayısal çıktı uydurulmadı.')
            payload=build_academic_docx(result,validator=validate_screen_result,title='Akademik '+page+' Raporu',figures=figures,apa7=True)
        history[page].update(result=result,docx=payload)
        st.markdown('### Akademik '+page+' Raporu')
        st.download_button('Word indir (.docx)',payload,'psikoanaliz_akademik_rapor.docx',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',key='academic_word',on_click='ignore')
    except (ValueError,TypeError,KeyError,AttributeError) as exc:
        history.pop(page,None);st.warning('Akademik Word hazırlanamadı: '+str(exc))
