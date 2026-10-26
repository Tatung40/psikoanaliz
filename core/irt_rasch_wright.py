"""Rasch Wright data from supported mirt APIs; no estimation or recentering."""
import io
import numpy as np
import pandas as pd
from core.irt_python_mirt_parameters import python_mirt_binary_parameters


def rasch_wright_tables(outcome,responses,*,item_names,identifiers=None,bins=30):
    import mirt
    if outcome.model!='Rasch' or not outcome.fit.converged:
        raise ValueError('Yakınsamış Python Rasch kestirimi gerekli.')
    x=np.asarray(responses,float);names=list(item_names)
    if x.ndim!=2 or x.shape!=(outcome.n_persons,outcome.n_items) or not np.isfinite(x).all() or not np.isin(x,[0,1]).all():
        raise ValueError('Wright haritası için eksiksiz 0/1 yanıtları gerekli.')
    if not isinstance(bins,int) or isinstance(bins,bool) or bins<1:raise ValueError('Geçersiz histogram bölmesi.')
    ids=pd.Series(list(identifiers) if identifiers is not None else np.arange(1,len(x)+1))
    if len(ids)!=len(x) or ids.isna().any() or ids.duplicated().any():raise ValueError('Kişi kimlikleri eksiksiz ve benzersiz olmalı.')
    fit=outcome.fit
    if (fit.latent_mean is not None and not np.array_equal(np.asarray(fit.latent_mean),[0.])) or (fit.latent_covariance is not None and not np.array_equal(np.asarray(fit.latent_covariance),[[1.]])):
        raise ValueError('Rasch N(0,1) ölçeği uyuşmuyor.')
    items=python_mirt_binary_parameters(fit,model='1PL',item_names=names)
    if not np.array_equal(items.a.to_numpy(),np.ones(len(names))):raise ValueError('Rasch eğimleri 1 olmalı.')
    sc=mirt.fscores(fit,x.astype(int),method='EAP',n_quadpts=61,person_ids=ids.tolist())
    theta=np.asarray(sc.theta,float).reshape(-1);sd=np.asarray(sc.standard_error,float).reshape(-1)
    if theta.shape!=(len(x),) or sd.shape!=theta.shape or not np.array_equal(sc.person_ids,ids) or not np.isfinite(np.r_[theta,sd,items.b]).all() or np.any(sd<=0):
        raise ValueError('Geçersiz kişi/madde kestirimleri.')
    people=pd.DataFrame({'Birey':ids,'θ':theta,'Posterior SD':sd})
    items=items[['Madde','b']].sort_values('b',kind='stable').reset_index(drop=True)
    lo=min(theta.min(),items.b.min())-.5;hi=max(theta.max(),items.b.max())+.5
    counts,edges=np.histogram(theta,bins=np.linspace(lo,hi,bins+1))
    histogram=pd.DataFrame({'Alt':edges[:-1],'Üst':edges[1:],'Merkez':(edges[:-1]+edges[1:])/2,'Kişi sayısı':counts})
    return dict(persons=people,items=items,histogram=histogram)


def wright_figure(tables):
    # Reuse the established Rasch plot; change its WLE legend to actual EAP.
    from ui.rasch_view import wright_map
    people,items,h=tables['persons'],tables['items'],tables['histogram']
    if people.empty or items.empty or not np.isfinite(np.r_[people['θ'],items.b]).all() or int(h['Kişi sayısı'].sum())!=len(people):
        raise ValueError('Geçersiz Wright tabloları.')
    fig=wright_map({'wright':{'persons':people['θ'].to_numpy(),'items':items.set_index('Madde').b}},bins=len(h))
    fig.data[0].name='Kişiler (EAP; N(0,1))'
    fig.update_layout(title='Rasch Wright Haritası — Python mirt; N(0,1)')
    return fig


def wright_exports(tables):
    from reports.irt_b14_report import safe_frame
    frames={name:safe_frame(frame) for name,frame in tables.items()}
    csv={name:frame.to_csv(index=False).encode('utf-8-sig') for name,frame in frames.items()}
    output=io.BytesIO()
    with pd.ExcelWriter(output,engine='openpyxl') as writer:
        for name,frame in frames.items():frame.to_excel(writer,index=False,sheet_name=name)
    return csv,output.getvalue()
