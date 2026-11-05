"""Frozen independent evidence, chronology and profile audits; zero calibration/MC."""
import ast
import io
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from validation.q1_q3.archive import unpack,restore
from validation.q1_q3.ci_mode import permitted

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'validation/q1_q3'


def load(path):return json.loads((ROOT/path).read_text())
def criteria(source):
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CRITERIA' for t in node.targets):
            return json.loads(json.dumps(ast.literal_eval(node.value)))
    raise AssertionError('Missing literal preregistered criteria')


def test_all_original_evidence_and_production_sources_remain_unchanged():
    manifest=load('validation/q1_q3/manifest.json')
    assert manifest['monte_carlo_executions']==manifest['new_certified_cells']==0
    for family in ('sha256','existing_evidence_sha256','source_sha256'):
        for path,digest in manifest[family].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    for r in load('validation/q1_q3/reference_sources.json'):
        assert r['identical']
        assert unpack(r['historical'])==(ROOT/r['path']).read_bytes()


@pytest.mark.parametrize('file,count',[('r_polytomous.json',4),('r_rsm.json',3),('r_34.json',2),('r_gt.json',5)])
def test_actual_r_archives_are_lossless_scoped_and_keep_failures(file,count):
    d=load('validation/evidence/'+file)
    assert len(d['cases'])==count and not d['certified'] and d['monte_carlo_executions']==0
    assert d['run'] and d['artifact'] and d['checkout']
    for case in d['cases']:
        if case.get('model')=='4PL':
            assert case['status']=='NOT VERIFIED'
            continue
        if 'R_raw' in case:
            raw=json.loads(unpack(case['R_raw']))
            assert raw['mirt']=='1.48' and raw['converged']
            assert case['status']=='NUMERICAL AGREEMENT / NOT CERTIFIED'
            assert all(case['checks'].values())
            data=pd.read_csv(io.BytesIO(unpack(case['data'])))
            assert data.shape[0]>0 and data.shape[1]>0
        else:
            assert case['comparison']['status']=='NUMERICALLY VALIDATED'
            for blob in case['raw'].values():unpack(blob)
    if file=='r_34.json':
        failure=d['historical_matched_4PL']
        raw=json.loads(unpack(failure['report']))
        assert raw['status']=='FAIL'
        assert raw['models'][0]['R']['converged'] is False
        assert 'NOT CERTIFIED' in failure['status']


def test_git_parent_chains_and_original_criteria_not_retrospective_thresholds():
    d=load('validation/q1_q3/preregistration.json')
    scripts=['validation/polytomous_mc_study.py','validation/bifactor_recovery_study.py']
    for record,script in zip(d['records'],scripts):
        assert record['status']=='VERIFIED GIT PREREGISTRATION'
        previous=record['criteria_commit']
        for node in record['ordering_proof']['parent_chain']:
            assert previous in node['parents']
            previous=node['sha']
        assert previous==record['results_commit']
        assert not record['criteria_changed_after_results']
        assert criteria(unpack(record['criteria_source']).decode())==record['criteria']
        assert criteria((ROOT/script).read_text())==record['criteria']
        for path,digest in record['results_sha256'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
            assert load(path)['criteria']==record['criteria']
    assert any('post-hoc method repair' in s for s in d['records'][0]['limitations'])


def test_profile_objective_reuses_actual_independent_r_probability_reference():
    from validation.r_mirt_phase1.equivalence_audit import evaluate,BOUNDS
    raw=load('validation/evidence/r_34.json')['historical_matched_4PL']
    data=pd.read_csv(ROOT/raw['fixture'])
    parameters=pd.read_csv(io.BytesIO(unpack(raw['R_parameters'])))
    p=parameters.pivot(index='item',columns='parameter',values='value').loc[data.columns,list(BOUNDS)]
    curves=pd.read_csv(io.BytesIO(unpack(raw['R_curves'])))
    probability=curves.loc[curves.category==1].pivot(index='theta',columns='item',values='probability').loc[:,data.columns]
    ours=evaluate(data.to_numpy(),p,probability.index.to_numpy(),np.ones(len(probability))/len(probability))[0]
    np.testing.assert_allclose(ours,probability.to_numpy(),atol=1e-10,rtol=0)


def test_fixed_coefficient_profiles_reproduce_strict_scores_without_refitting():
    from validation.q1_q3.profile import gh
    from validation.r_mirt_phase1.equivalence_audit import evaluate,BOUNDS
    from validation.r_mirt_phase1.m12_diagnostic import scores,feasible_scores
    data=pd.read_csv(ROOT/'tests/fixtures/irt_b11/data_irt4pl_ok.csv')
    profile=load('validation/q1_q3/profile_polished.json')
    assert len(profile['points'])==5 and not profile['independent_R_optimizer_executed']
    assert profile['monte_carlo_executions']==0
    z,w=gh(61);az,aw=gh(101)
    for point in profile['points']:
        p=pd.DataFrame.from_dict(point['parameters'],orient='index').loc[data.columns,list(BOUNDS)]
        assert p.loc['M12','d']==point['u']
        assert all(p[k].between(lo,hi).all() for k,(lo,hi) in BOUNDS.items())
        g=feasible_scores(p,scores(data.to_numpy(),p,z,w));g.loc['M12','d']=0
        assert np.abs(g.to_numpy()).max()<=1e-5
        assert point['stationary_GH61']
        a=feasible_scores(p,scores(data.to_numpy(),p,az,aw));a.loc['M12','d']=0
        assert not point['stationary_GH101'] and np.abs(a.to_numpy()).max()>1e-5
        assert evaluate(data.to_numpy(),p,z,w)[2]==pytest.approx(point['logLik_GH61'],abs=1e-8)
        assert all(r['information_min']>0 and r['condition']<=1e8 for r in point['history'])


def test_failed_optimizer_flags_are_not_promoted_to_stationary_profiles():
    for path in ('profile_uncentered.json','profile_centered.json'):
        d=load('validation/q1_q3/'+path)
        assert len(d['points'])==10
        assert not any(p['stationary_GH61'] for p in d['points'])
        assert all(p['nuisance_score_max']>1e-5 for p in d['points'])
    protocol=load('validation/q1_q3/PROFILE_PROTOCOL.json')
    assert len(protocol['grid'])<=25 and not protocol['monte_carlo']


def test_no_mc_ci_mode_fails_closed_for_production_or_unrelated_changes():
    assert permitted(['validation/q1_q3/profile.py','validation/evidence/r_34.json'])
    for path in ('core/irt_binary_inference.py','app.py','validation/polytomous_mc_study.py'):
        assert not permitted(['validation/evidence/r_34.json',path])
    assert not permitted([])


def test_original_r_inputs_and_outputs_restore_without_network_or_refits(tmp_path):
    restore(tmp_path)
    for file,base in [('r_polytomous.json','3/r-mirt-polytomous'),('r_rsm.json','0/r-mirt-rsm'),('r_34.json','0/r-mirt-34')]:
        archive=load('validation/evidence/'+file)
        assert (tmp_path/base/'report.json').read_bytes()==unpack(archive['original_report'])
        for case in archive['cases']:
            if 'data' not in case:continue
            path=tmp_path/base/case.get('model',case.get('case'))
            assert (path/'data.csv').read_bytes()==unpack(case['data'])
            assert (path/'r_result.json').read_bytes()==unpack(case['R_raw'])
            assert len(pd.read_csv(path/'theta.csv'))==81
    raw=load('validation/evidence/r_34.json')['historical_matched_4PL']
    assert hashlib.sha256((ROOT/raw['fixture']).read_bytes()).hexdigest()==raw['fixture_sha256']
    assert (tmp_path/'2/refit/4PL/R_96wsqzr8/parameters.csv').read_bytes()==unpack(raw['R_parameters'])
