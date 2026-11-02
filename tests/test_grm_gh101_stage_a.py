"""Tiny no-optimization fixtures only; CI never launches Stage A."""
import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from validation.grm_gh101_stage_a import statistics as S
from validation.grm_gh101_stage_a import audit as A
from validation.grm_gh101_stage_a import runner as R


def fixture():
    rows=[];ordinal={}
    for s in range(2026110100,2026110200):
        for impact in ('minus','zero'):
            event=(s%7==0 if impact=='minus' else s%11==0)
            d=dict(valid=True,holm_event=event,bh_event=event,raw_rejections=[int(event),0])
            rows.append(dict(seed=s,impact=impact,valid=True,decisions=d))
            ordinal[s,impact]=dict(methods={m:dict(valid=True,holm_event=(s%(3 if m=='total' else 5)==0)) for m in ('total','rest')})
    return rows,ordinal


def test_fixed_protocol_before_outcomes():
    p=json.loads((R.HERE/'protocol.json').read_text())
    assert p['seeds']==list(range(2026110100,2026110200))
    assert p['analysis_order']==[dict(seed=s,impact=k) for s in p['seeds'] for k in ('minus','zero')]
    assert p['planned_analyses']==200 and p['allowed_stages']==[100]
    assert p['gates']['audit_score_lte']==1e-5 and not p['pilot_reuse']['allowed']


def test_native_cold_start_without_real_optimizer(monkeypatch):
    x=np.tile(np.arange(4).repeat(25)[:,None],(2,10));g=np.repeat([0,1],100)
    inputs=R.G.validate_inputs(__import__('pandas').DataFrame(x,columns=[f'P{i:02d}' for i in range(1,11)]),g,0,1,[f'P{i:02d}' for i in range(1,9)],categories=4)
    designs,_=R.layouts();calls=[]
    def fake(fn,x0,**kwargs):
        calls.append(x0.copy());return SimpleNamespace(x=x0,nit=0)
    monkeypatch.setattr(R.G,'minimize',fake)
    for value in designs.values():
        d=value['design'];expected,_=R.cold_start(d,inputs)
        f=R.G.Fitter(inputs,points=101);f.audit_grid=R.D.reference_grid(161)
        f.fit(d.roles,start_fit=None,information=False)
        np.testing.assert_array_equal(calls[-1],expected)
    assert len(calls)==7


def test_statistics_independent_oracle_and_missing_outcomes():
    rows,ordinal=fixture();rows=rows[:-3];rows[0].update(valid=False,decisions=S.decisions({'valid':False}))
    ordinal[2026110101,'minus']['methods']['rest']['valid']=False
    summary=S.summarize(rows,ordinal);A.statistical_audit(rows,ordinal,summary)
    assert summary['attempted']==197 and summary['invalid']==1 and summary['not_run']==3
    assert summary['contrasts']['GRM_zero-minus-minus']['n']==97
    assert summary['contrasts']['total-minus-rest_minus']['n']==99
    broken=copy.deepcopy(summary);broken['conditions']['minus']['holm_fwer']['mcse']=10
    with pytest.raises(AssertionError):A.statistical_audit(rows,ordinal,broken)


def test_invalid_family_suppression_and_lr_p_decisions():
    assert S.decisions({'valid':False})['p'] is None
    row={'valid':True,'lr':[dict(item=item,test=test,df=df,valid=True,lr={'161':0.,'101':0.}) for item in ('P09','P10') for test,df in [('omnibus',4),('thresholds',3),('slope',1)]]}
    d=S.decisions(row);assert d['p']==[1.,1.] and not d['holm_event']
    assert not d['bh_event']
    row['lr'].pop()
    with pytest.raises(ValueError):S.decisions(row)


def test_sha_tampering_fails_before_audit(tmp_path):
    (tmp_path/'metadata.json').write_text('{}')
    (tmp_path/'summary.json').write_text(json.dumps({'records_sha256':'wrong','records_gzip_sha256':'wrong'}))
    (tmp_path/'records.jsonl.gz').write_bytes(gzip.compress(b''))
    with pytest.raises(AssertionError):A.audit(tmp_path)


def test_duplicate_attempts_never_pooled():
    rows,ordinal=fixture()
    with pytest.raises(ValueError):S.summarize(rows+[rows[0]],ordinal)
