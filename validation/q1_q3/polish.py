"""Finite bounded score corrections under the separately committed polish protocol."""
import argparse
import hashlib
import json
import time
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd

from validation.q1_q3.profile import ROOT,HERE,gh
from validation.r_mirt_phase1.equivalence_audit import evaluate,BOUNDS
from validation.r_mirt_phase1.m12_diagnostic import scores,feasible_scores
from validation.r_mirt_phase1.information_audit import information


def run(commit):
    protocol=json.loads((HERE/'PROFILE_POLISH_PROTOCOL.json').read_text())
    original=json.loads((HERE/'profile_centered.json').read_text())
    x=pd.read_csv(ROOT/'tests/fixtures/irt_b11/data_irt4pl_ok.csv')
    z,w=gh(61);az,aw=gh(101)
    output=dict(protocol_commit=commit,source_sha256=hashlib.sha256((HERE/'profile_centered.json').read_bytes()).hexdigest(),
                status='DIAGNOSTIC / NOT CERTIFIED',independent_R_optimizer_executed=False,
                monte_carlo_executions=0,
                versions={k:importlib.metadata.version(k) for k in ('numpy','scipy','pandas','mirt')},points=[])
    for u in protocol['grid']:
        started=time.monotonic()
        source=next(r for r in original['points'] if r['u']==u)
        p=pd.DataFrame.from_dict(source['parameters'],orient='index').loc[x.columns,list(BOUNDS)]
        fixed=list(p.index).index('M12')*4+3
        def score(v):return scores(x.to_numpy(),v,z,w)
        def value(v):return evaluate(x.to_numpy(),v,z,w)[2]
        def projected(v):
            r=feasible_scores(v,score(v)).to_numpy().ravel();r[fixed]=0
            return float(np.abs(r).max())
        history=[]
        for step in range(3):
            g=score(p).to_numpy().ravel();v=p.to_numpy().ravel()
            blocked=[j for j,q in enumerate(v) if j==fixed or
                     (q==list(BOUNDS.values())[j%4][0] and g[j]<=0) or
                     (q==list(BOUNDS.values())[j%4][1] and g[j]>=0)]
            free=[j for j in range(len(v)) if j not in blocked]
            h,_=information(score,p,float(np.cbrt(np.finfo(float).eps)))
            h=(h+h.T)/2;face=h[np.ix_(free,free)]
            eig=np.linalg.eigvalsh(face);condition=float(np.linalg.cond(face))
            record=dict(step=step,score=projected(p),information_min=float(eig[0]),condition=condition)
            history.append(record)
            if eig[0]<=0 or condition>1e8:record['stop']='INFORMATION GATE';break
            if projected(p)<=1e-5:record['stop']='STATIONARY';break
            direction=np.zeros(len(v));direction[free]=np.linalg.solve(face,g[free])
            accepted=False;old=value(p)
            for attempt in range(10):
                nv=v+direction*(.5**attempt)
                if any(nv[j]<list(BOUNDS.values())[j%4][0] or nv[j]>list(BOUNDS.values())[j%4][1]
                       for j in range(len(nv))):continue
                candidate=pd.DataFrame(nv.reshape(p.shape),index=p.index,columns=p.columns)
                if value(candidate)>=old and projected(candidate)<projected(p):
                    p=candidate;accepted=True;record['accepted_backtrack']=attempt;break
            if not accepted:record['stop']='NO MONOTONE SCORE-IMPROVING STEP';break
        raw=projected(p)
        ag=feasible_scores(p,scores(x.to_numpy(),p,az,aw)).to_numpy().ravel();ag[fixed]=0
        largest=int(np.argmax(np.abs(ag)))
        output['points'].append(dict(u=u,history=history,seconds=time.monotonic()-started,
            logLik_GH61=value(p),logLik_GH101=evaluate(x.to_numpy(),p,az,aw)[2],
            nuisance_score_max=raw,audit_nuisance_score_max=float(np.abs(ag).max()),
            stationary_GH61=bool(raw<=1e-5),stationary_GH101=bool(np.abs(ag).max()<=1e-5),
            audit_largest_component=f'{p.index[largest//4]}:{p.columns[largest%4]}',
            audit_largest_score=float(ag[largest]),
            M12=p.loc['M12'].to_dict(),parameters=p.to_dict('index'),
            boundaries=[f'{i}:{k}' for i in p.index for k in p.columns if p.loc[i,k] in BOUNDS[k]]))
        (HERE/'profile_polished.json').write_text(json.dumps(output,indent=2)+'\n')
        print(json.dumps({k:output['points'][-1][k] for k in ('u','seconds','nuisance_score_max','stationary_GH61')}),flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol-commit',required=True)
    run(parser.parse_args().protocol_commit)
