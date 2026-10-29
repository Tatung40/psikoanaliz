"""Lazy, deterministic teaching data. Generators never certify an estimator."""
from dataclasses import dataclass, asdict
from functools import lru_cache
import io
import json

import numpy as np
import pandas as pd

from core.generalizability import DESIGNS
from core.scoring import score_dichotomous
from core.data_processing import load_dataset
from reports.excel_export import sanitize_excel_cell

LEVELS = ('Temel', 'Gerçekçi', 'İleri')
IRT_MODELS = ('Rasch','1PL','2PL','3PL','4PL','GRM','PCM','GPCM','NRM','RSM')
G_PAGE = 'Genellenebilirlik Kuramı (G ve D Çalışmaları)'
M_PAGE = 'Çok Boyutlu MTK (MIRT)'
MISSING_PAGE = 'Eksik Veri ve Atama'


@dataclass(frozen=True)
class Dataset:
    id: str
    family: str
    model: str
    level: int
    n: int
    k: int
    seed: int
    design: str = ''
    dimensions: int = 1
    effect: str = ''

    @property
    def title(self):
        name = self.design or self.model
        return f'{name} — {LEVELS[self.level]} (N={self.n}, k={self.k})'


@lru_cache(maxsize=1)
def registry():
    from core.irt_multidim import MODELS
    from core.irt_multidim_factorized import MODELS as BLOCKS
    from core.irt_multidim_rsm_stack import MODELS as RSM
    from core.irt_multidim_general_report import MODELS as GENERAL
    entries=[]
    def add(family,model,level,n,k,**kwargs):
        idx=len(entries)
        entries.append(Dataset(f'{family}_{idx:03d}',family,model,level,n,k,20261028+idx,**kwargs))
    for model in IRT_MODELS:
        for level,(n,k) in enumerate(((300,8),(500,12),(800,16))):
            add('irt',model,level,n,k)
    for model in (*MODELS,*BLOCKS,*RSM,*GENERAL):
        for level,(n,k,dims) in enumerate(((300,8,2),(500,12,3),(700,16,2))):
            # Free-asymptote general profiles explicitly support 2D only.
            if 'serbest' in model:dims=2
            # Bifactor fit requires 3–4 specific groups with >= 3 items each; 4 per group converges at screen defaults.
            if model=='Bifactor':dims,k=4,max(k,12)
            add('mirt',model,level,n,k,dimensions=dims)
    for model in ('2PL','GRM','PCM','GPCM','RSM','NRM'):
        for level,effect in enumerate(('null','uniform','nonuniform')):
            # RSM/PCM have fixed a; advanced scenario uses a different location effect.
            if model in ('PCM','RSM') and effect=='nonuniform':effect='location_strong'
            add('dif',model,level,600,8,effect=effect)
    for design in DESIGNS:
        for level,(p,i,r) in enumerate(((6,3,2),(12,4,3),(20,5,4))):
            facets=DESIGNS[design]; n=p*(i if 'i' in facets else 1)*(r if 'r' in facets else 1)
            add('g',design,level,n,len(facets),design=design)
    for level,(n,k) in enumerate(((100,4),(300,20),(500,50))):add('abc','ABCDE',level,n,k)
    for level in range(3):
        add('factor','Sürekli iki faktör',level,(150,300,500)[level],(8,12,16)[level],dimensions=2)
        add('missing','MCAR öğretim verisi',level,(100,200,300)[level],8)
    return {d.id:d for d in entries}


def analysis_catalog():
    """Actual routed screens; non-analysis help/home pages deliberately excluded."""
    common=['3. Puanlama & Veri Hazırlama','3B. Betimsel İstatistikler','4. Test İstatistikleri',
        '5. Madde Analizi','6. Çeldirici Analizi','7. Güvenirlik & Ölçme Hatası',
        '8. Grup Bazlı Analizler','9. Bilimsel Grafikler','10. Raporlama','KTK: Çift Motor Rapor',
        'MTK: Rasch','MTK: 1PL','MTK: 2PL','MTK: 3PL','MTK: 4PL','MTK: Karşılaştırma',
        'MTK: Varsayım Denetimi','MTK: Çok Modelli DIF Taraması',
        'MTK: Çok Kategorili B14','MTK: RSM','MTK: Python mirt DIF',
        'MTK: GRM Formel DIF','MTK: Çok Kategorili Formel DIF','MTK: Gözlenen Puan DIF',
        'MTK: 3PL/4PL Bootstrap DIF','Boyutluluk ve Faktör Analizi',
        'Yerel Bağımsızlık ve Madde Uyumu',G_PAGE,M_PAGE,MISSING_PAGE]
    common += [f'MTK: Python mirt {m}' for m in IRT_MODELS if m!='RSM']
    return tuple(common)


def available(page):
    datasets=list(registry().values())
    if page in ('2. Veri Yükleme','1. Ana Sayfa'):return datasets
    if page==G_PAGE:return [d for d in datasets if d.family=='g']
    if page==M_PAGE:return [d for d in datasets if d.family=='mirt']
    if page==MISSING_PAGE:return [d for d in datasets if d.family=='missing']
    if 'DIF' in page:
        choices=[d for d in datasets if d.family=='dif']
        if page=='MTK: GRM Formel DIF':choices=[d for d in choices if d.model=='GRM']
        if page=='MTK: Python mirt DIF':choices=[d for d in choices if d.model!='RSM']
        if page in ('MTK: 3PL/4PL Bootstrap DIF','MTK: Çok Modelli DIF Taraması'):choices=[d for d in choices if d.model=='2PL']
        if page=='MTK: Çok Kategorili Formel DIF':choices=[d for d in choices if d.model in ('PCM','GPCM','RSM','NRM')]
        if page=='MTK: Gözlenen Puan DIF':choices=[d for d in choices if d.model in ('2PL','GRM')]
        return choices
    if page=='6. Çeldirici Analizi' or page=='3. Puanlama & Veri Hazırlama':
        return [d for d in datasets if d.family=='abc']
    model=page.removeprefix('MTK: Python mirt ').removeprefix('MTK: ')
    if model in IRT_MODELS:return [d for d in datasets if d.family=='irt' and d.model==model]
    if page=='MTK: Çok Kategorili B14':return [d for d in datasets if d.family=='irt' and d.model in ('GRM','PCM','GPCM','NRM')]
    if page=='Boyutluluk ve Faktör Analizi':return [d for d in datasets if d.family=='factor' or d.family=='irt' and d.model in ('2PL','GRM')]
    if page in ('Yerel Bağımsızlık ve Madde Uyumu','MTK: Karşılaştırma'):
        return [d for d in datasets if d.family=='abc' or d.family=='irt' and d.model=='2PL']
    return [d for d in datasets if d.family in ('abc',) or d.family=='irt' and d.model in ('2PL','GRM')] if page in analysis_catalog() else []


def _parameters(spec,k,dims=1):
    model=spec.model.split(' ')[0]
    if model=='1PL':model='1PL'
    if model=='Bifactor':model='2PL'
    if model=='Rasch':model='1PL'
    binary=model in ('1PL','2PL','3PL','4PL')
    a=np.linspace(.9,1.5,k)
    if spec.model.startswith(('Rasch','1PL','PCM','RSM')):a[:]=1
    if spec.model=='1PL':a[:]=1.2
    if spec.model=='1PL':model='2PL'  # common a=1.2; simdata('1PL') fixes a=1.
    b=np.linspace(-1.2,1.2,k)
    params={'discrimination':a,'difficulty':b}
    if model in ('3PL','4PL'):params['guessing']=np.full(k,.15)
    if model=='4PL':params['upper']=np.full(k,.9)
    if not binary:
        if model=='GRM':params['thresholds']=b[:,None]+np.array([-1.,0.,1.])
        elif model in ('PCM','GPCM'):
            params['steps']=b[:,None]+np.array([-.8,0.,.8])
            params.pop('difficulty')
        elif model=='RSM':params['thresholds']=np.tile([-.8,0.,.8],(k,1))
        elif model=='NRM':
            params={'slopes':np.tile([0.,1.,-.6,.5],(k,1)),
                    'intercepts':np.column_stack([np.zeros(k),-b,.3*b,-.3*b])}
    return model,params


def materialize(dataset_id):
    """Return fresh frames: edits never mutate a cached/shared dataset."""
    return _generate(dataset_id)


def _generate(dataset_id):
    import mirt
    spec=registry()[dataset_id];rng=np.random.default_rng(spec.seed)
    metadata={**asdict(spec),'synthetic':True,'status':'GENERATED / schema checked; calibration not certified',
              'generator_version':'3','numpy':np.__version__,'mirt':'1.2.0',
              'response_rng':'Independent SeedSequence([call_seed, 104729]); group/item call_seed offsets in generator',
              'latent_distribution':'N(0,I); independent factors',
              'category_semantics':'NRM nominal; binary 0/1; ordinal 0–3; G/factor metric score'}
    key={}; settings={}; group='grup'; person='ogrenci_id'
    if spec.family=='g':
        p,i,r=((6,3,2),(12,4,3),(20,5,4))[spec.level]
        facets=DESIGNS[spec.design]; n_i=i if 'i' in facets else 1;n_r=r if 'r' in facets else 1
        vp=rng.normal(0,3,p);vi=rng.normal(0,.6+spec.level*.2,n_i);vr=rng.normal(0,.5,n_r)
        # Distinct nested IDs must also have independent child effects per parent.
        if spec.design=='p x (r:i)':vr=rng.normal(0,.5,(n_i,n_r))
        if spec.design=='p x (i:r)':vi=rng.normal(0,.6+spec.level*.2,(n_r,n_i))
        records=[]
        for pp in range(p):
            for ii in range(n_i):
                for rr in range(n_r):
                    item_effect=vi[rr,ii] if vi.ndim==2 else vi[ii]
                    rater_effect=vr[ii,rr] if vr.ndim==2 else vr[rr]
                    row={'p':f'P{pp+1:03d}','score':20+vp[pp]+item_effect+rater_effect+rng.normal(0,.6)}
                    if 'i' in facets:row['i']=f'I{ii+1}' if spec.design!='p x (i:r)' else f'R{rr+1}_I{ii+1}'
                    if 'r' in facets:row['r']=f'R{rr+1}' if spec.design!='p x (r:i)' else f'I{ii+1}_R{rr+1}'
                    records.append(row)
        df=pd.DataFrame(records)[[*facets,'score']];items=['score'];group=None;person='p'
        counts={f:{'p':p,'i':i,'r':r}[f] for f in facets}
        scenarios={f:[n] if f=='p' else [n,n+1] for f,n in counts.items()}
        settings={'gt_design':spec.design,'gt_format':'Uzun','gt_score':'score',
                  **{'gt_col_'+f:f for f in facets},
                  **{f'gt_d_{spec.design}_{f}':','.join(map(str,v)) for f,v in scenarios.items() if f!='p'}}
        metadata.update(generative_sd={'p':3.,'i':.6+spec.level*.2,'r':.5,'residual':.6},
                        counts=counts,scenarios=scenarios,columns={f:f for f in facets},
                        realized_effects={'p':vp.tolist(),'i':vi.tolist(),'r':vr.tolist()},
                        status='LIBRARY TESTED — all 15 G/D examples exercised by regression; NOT CERTIFIED')
    elif spec.family=='factor':
        z=rng.normal(size=(spec.n,2));loads=np.zeros((spec.k,2));loads[:spec.k//2,0]=.8;loads[spec.k//2:,1]=.8
        df=pd.DataFrame(z@loads.T+rng.normal(0,.6,(spec.n,spec.k)),columns=[f'M{j+1:02d}' for j in range(spec.k)])
        items=list(df);metadata.update(loadings=loads.tolist(),residual_sd=.6)
    else:
        dims=spec.dimensions;k=spec.k;theta=rng.normal(size=(spec.n,dims))
        model,params=_parameters(spec,k,dims)
        if spec.family in ('abc','missing'):model='2PL';params={'discrimination':np.linspace(.8,1.5,k),'difficulty':np.linspace(-1.3,1.3,k)}
        groups=np.where(np.arange(spec.n)<spec.n//2,'Referans','Odak')
        if spec.family=='dif':
            # Same N(0,1) population; M03 contains the sole planted effect.
            parts=[]
            for g in ('Referans','Odak'):
                par={k:np.array(v,copy=True) for k,v in params.items()}
                if g=='Odak' and spec.effect!='null':
                    if model=='NRM':
                        par['slopes' if spec.effect=='nonuniform' else 'intercepts'][2,1]+=.6
                    elif spec.effect=='nonuniform':par['discrimination'][2]*=1.5
                    elif model=='GRM':par['thresholds'][2]+=.6
                    elif model in ('PCM','GPCM'):par['steps'][2]+=.6 if spec.effect=='uniform' else 1.
                    else:par['difficulty'][2]+=.6 if spec.effect=='uniform' else 1.
                parts.append(_simulate(model,spec.n//2,k,theta[np.flatnonzero(groups==g)],par,spec.seed+(g=='Odak')))
            x=np.vstack(parts);metadata.update(effect_item='M03',anchors=['M01','M02'],effect=spec.effect,
                effect_definition='Odak: b/steps/threshold +0.6; strong location +1; slope ×1.5; NRM category B +0.6. Null: unchanged.')
            settings={'pmd_model':spec.model,'pmd_anchors':['M01','M02'],'pmd_studied':['M03'],
                      'poly_model':spec.model if spec.model in ('PCM','GPCM','RSM','NRM') else 'PCM',
                      'poly_anchors':['M01','M02'],'grm_anchors':['M01','M02']}
        elif spec.family=='mirt':
            # Public simulation API; simple structure, no new calibration algorithm.
            block=np.minimum(np.arange(k)*dims//k,dims-1)
            if spec.model=='Bifactor':
                loads=np.zeros((k,dims));loads[:,0]=np.linspace(.7,1.2,k)
                loads[np.arange(k),1+np.arange(k)*(dims-1)//k]=np.resize([.5,.9,.65,.8],k)
            else:
                loads=np.zeros((k,dims));loads[np.arange(k),block]=params.get('discrimination',np.ones(k))
            x=np.empty((spec.n,k),int)
            # Scalar projections also support RSM and factorized constrained profiles.
            for j in range(k):
                par={name:v[j:j+1] for name,v in params.items()}
                if model!='NRM':par['discrimination']=np.ones(1)
                x[:,j]=_simulate(model,spec.n,1,theta@loads[j],par,spec.seed+j)[:,0]
            settings={'md_model':spec.model,'md_dims':dims,'md_corr':'Tümü' if '(SciPy)' in spec.model else 'Ortogonal',
                      'md_fixed_c':.15,'md_fixed_u':.9}
            metadata.update(loadings=loads.tolist(),covariance=np.eye(dims).tolist(),
                            parameter_note='Scalar theta projection through loadings; ordinal/binary scalar discrimination=1; NRM keeps category slopes.')
        else:
            x=_simulate(model,spec.n,k,theta[:,0],params,spec.seed)
        items=[f'M{j+1:02d}' for j in range(k)]
        if model=='NRM':x=np.array(list('ABCD'))[x]
        df=pd.DataFrame(x,columns=items)
        metadata['parameters']={k:np.asarray(v).tolist() for k,v in params.items()}
        if spec.family=='abc':
            key=dict(zip(items,np.resize(list('ACEBD'),spec.k)))
            raw=np.empty_like(x,dtype='<U1')
            for j,item in enumerate(items):
                wrong=[v for v in 'ABCDE' if v!=key[item]]
                raw[:,j]=np.where(x[:,j]==1,key[item],rng.choice(wrong,spec.n))
            df=pd.DataFrame(raw,columns=items)
            if spec.k==4:
                key=dict(zip(items,'ACEB'))
                df.iloc[:3]=[['A','C','E','B'],['B','C','D','A'],['A','D','E','B']]
                metadata['fixed_teaching_rows']=3  # Prescribed illustration, not a pure IRT calibration fixture.
        if spec.family=='missing':
            mask=rng.random(df.shape)<(.03,.10,.20)[spec.level]
            df=df.astype(float).mask(mask);metadata.update(mechanism='MCAR by independent Bernoulli mask',missing_rate=(.03,.10,.20)[spec.level])
        if model=='NRM':
            settings.update(python_mirt_nominal_labels='A,B,C,D',pmd_labels='A,B,C,D',md_labels='A,B,C,D')
    if spec.family!='g':
        df.insert(0,person,[f'O{i+1:04d}' for i in range(len(df))])
        df[group]=np.where(np.arange(len(df))<len(df)//2,'Referans','Odak')
    return {'data':df,'items':items,'id':person,'group':group,'key':key,
            'settings':settings,'metadata':metadata,'spec':spec}


def _simulate(model,n,k,theta,params,seed):
    import mirt
    # Do not restart the latent-theta RNG for the uniform response draws.
    response_seed=int(np.random.SeedSequence([int(seed),104729]).generate_state(1)[0])
    if model=='RSM':
        from mirt.models import RatingScaleModel
        item=RatingScaleModel(k,4).set_parameters(difficulty=params['difficulty'],thresholds=params['thresholds'][0])
        return mirt.simdata(item,n_persons=n,theta=theta,seed=response_seed)
    return mirt.simdata(model,n_persons=n,n_items=k,n_categories=4,theta=theta,seed=response_seed,**params)


def scored(bundle):
    if not bundle['key']:return bundle['data'].copy()
    data,_=score_dichotomous(bundle['data'],bundle['items'],bundle['key'],id_column=bundle['id'])
    if bundle['group']:data[bundle['group']]=bundle['data'][bundle['group']]
    return data


def downloads(bundle):
    frames={'veri':bundle['data']}
    if bundle['key']:
        frames['cevap_anahtari']=pd.DataFrame({'madde':list(bundle['key']),'dogru_cevap':list(bundle['key'].values())})
        frames['puanlanmis']=scored(bundle)
    if bundle['spec'].design=='p x i':
        frames['genis']=bundle['data'].pivot(index='p',columns='i',values='score').reset_index()
    files={}
    for name,frame in frames.items():
        safe=frame.map(sanitize_excel_cell)
        for ext,sep in [('csv',','),('tsv','\t'),('txt','\t')]:
            files[f'{name}.{ext}']=safe.to_csv(index=False,sep=sep).encode('utf-8-sig')
        out=io.BytesIO()
        with pd.ExcelWriter(out,engine='openpyxl') as w:safe.to_excel(w,index=False,sheet_name=name)
        files[f'{name}.xlsx']=out.getvalue()
    out=io.BytesIO()
    with pd.ExcelWriter(out,engine='openpyxl') as w:
        for name,frame in frames.items():frame.map(sanitize_excel_cell).to_excel(w,index=False,sheet_name=name)
    files['tum_sayfalar.xlsx']=out.getvalue()
    files['uretim_bilgileri.json']=json.dumps(bundle['metadata'],ensure_ascii=False,indent=2).encode()
    return files


def input_frame(bundle):
    """Exercise the real CSV loader, not a second import implementation."""
    frame,_=load_dataset(bundle['data'].to_csv(index=False).encode('utf-8-sig'),file_type='csv',delimiter=',')
    return frame
