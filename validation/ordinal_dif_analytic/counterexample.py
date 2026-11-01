"""Deterministic category/score integration only. No simulation or estimation API."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'validation/ordinal_dif_paired/evidence/mechanism.json'


def probabilities(theta,a,b):
    survival=expit(np.asarray(a)[None,:,None]*(np.asarray(theta)[:,None,None]-np.asarray(b)[None,:,:]))
    return np.diff(np.concatenate([np.zeros((*survival.shape[:2],1)),1-survival,np.ones((*survival.shape[:2],1))],axis=2),axis=2)


def joint_score(probs,weights,target,score_items):
    """P(score=s,Y_target=k), allowing target inclusion or exclusion explicitly."""
    probs=np.asarray(probs,float);weights=np.asarray(weights,float);items=list(score_items)
    C=probs.shape[2];joint=np.zeros(((C-1)*len(items)+1,C))
    if len(set(items))!=len(items) or target<0 or target>=probs.shape[1]:raise ValueError('Invalid item mapping')
    for q,w in enumerate(weights):
        other=np.ones(1)
        for i in items:
            if i!=target:other=np.convolve(other,probs[q,i])
        for category in range(C):
            offset=category if target in items else 0
            joint[offset:offset+len(other),category]+=w*probs[q,target,category]*other
    return joint


def table(points,impact,target,score_items):
    z,w=hermegauss(points);w=w/np.sqrt(2*np.pi)
    a=np.linspace(.9,1.8,10);b=np.linspace(-1,1,10)[:,None]+np.array([-1.,0.,1.])
    groups=[]
    for mu in (0.,impact):
        joint=joint_score(probabilities(z+mu,a,b),w,target,score_items);mass=joint.sum(1)
        conditional=joint/mass[:,None]
        groups.append(dict(score_probability=mass.tolist(),category_probability=conditional.tolist(),
                           conditional_mean=(conditional@np.arange(4)).tolist()))
    delta=np.array(groups[1]['category_probability'])-np.array(groups[0]['category_probability'])
    means=np.array(groups[1]['conditional_mean'])-np.array(groups[0]['conditional_mean'])
    score,category=np.unravel_index(np.argmax(np.abs(delta)),delta.shape)
    mean_score=int(np.argmax(np.abs(means)))
    return dict(target=f'P{target+1:02d}',score_items=[f'P{i+1:02d}' for i in score_items],groups=groups,
        category_difference=delta.tolist(),mean_difference=means.tolist(),
        largest_category_contrast=dict(score=int(score),category=int(category),difference=float(delta[score,category])),
        largest_mean_contrast=dict(score=mean_score,difference=float(means[mean_score]),
            category_differences=delta[mean_score].tolist(),
            group_score_probabilities=[g['score_probability'][mean_score] for g in groups]))


def compute():
    import platform
    from importlib.metadata import version
    cached=json.loads(CACHE.read_text());results={};checks=[]
    for points in (61,81):
        for impact in (0.,-.5):
            key=f'{points}_{impact}';results[key]={}
            for target in (8,9):
                item=f'P{target+1:02d}';results[key][item]={}
                for score,indices in [('total',list(range(10))),('rest',[i for i in range(10) if i!=target]),('anchors',list(range(8)))]:
                    result=table(points,impact,target,indices);results[key][item][score]=result
                    if score=='total':
                        old=cached[key][item]
                        checks.append(dict(grid=key,item=item,
                            cached_mean_max_error=float(np.max(np.abs(np.array(result['mean_difference'])-old['mean_difference']))),
                            cached_score_mass_max_error=max(float(np.max(np.abs(np.array(new['score_probability'])-prior['total_probability']))) for new,prior in zip(result['groups'],old['groups']))))
    stability={}
    for impact in (0.,-.5):
        for item in ('P09','P10'):
            for score in ('total','rest','anchors'):
                a=results[f'61_{impact}'][item][score];b=results[f'81_{impact}'][item][score]
                stability[f'{impact}_{item}_{score}']=dict(
                    max_category_step_difference=max(float(np.max(np.abs(np.array(x['category_probability'])-y['category_probability']))) for x,y in zip(a['groups'],b['groups'])),
                    max_contrast_step_difference=float(np.max(np.abs(np.array(a['category_difference'])-b['category_difference']))))
    return dict(execution='DETERMINISTIC CONVOLUTION/INTEGRATION ONLY; NO FITS OR RANDOM DATA',
        cache_sha256=hashlib.sha256(CACHE.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=version('scipy')),
        item_parameters=dict(a=np.linspace(.9,1.8,10).tolist(),b=(np.linspace(-1,1,10)[:,None]+[-1.,0.,1.]).tolist(),category_codes=[0,1,2,3]),
        assumptions='Known unidimensional GRM, local independence, D=1, identical a/b, normal latent distributions with mean0 versus impact and unit SD. Ordered categories0..3.',
        results=results,cached_checks=checks,stability=stability,
        scientific_status='NUMERICAL COUNTEREXAMPLE; NOT METHOD CERTIFICATION')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.write_text(json.dumps(compute(),ensure_ascii=False,separators=(',',':')))
