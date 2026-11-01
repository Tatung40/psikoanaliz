"""Small deterministic control/CRN checks; no calibration or large simulation."""
import hashlib
import json

import numpy as np
import pytest

from validation.dif_validation_study import gen_ordinal
from validation.ordinal_dif_paired import artifacts
from validation.ordinal_dif_paired.control import control_data,pair_contrast


def test_control_uses_identical_normals_and_uniforms_not_only_same_seed():
    old,g=gen_ordinal(12,14,-.5,False,29)
    x,groups,proof=control_data(29,old,12,14)
    np.testing.assert_array_equal(groups,g)
    np.testing.assert_array_equal(x.iloc[:12],old.iloc[:12])
    assert np.all(x.iloc[12:].to_numpy()>=old.iloc[12:].to_numpy())
    assert proof['reconstructed_negative_exactly'] and len(proof['calls'])==12
    bad=old.copy();bad.iloc[0,0]=(bad.iloc[0,0]+1)%4
    with pytest.raises(ValueError,match='reconstruct'):
        control_data(29,bad,12,14)


def test_paired_uncertainty_uses_actual_discordances_and_missing_pairs():
    row=pair_contrast([1,0,1],[0,1,1],4)
    assert row['differences']==[1.,-1.,0.]
    assert row['mean']==0 and row['mcse']==pytest.approx(1/np.sqrt(3))
    assert row['a_greater']==row['b_greater']==1
    assert row['missing_pair_range']==[-.25,.25]
    constant=pair_contrast([0]*10,[0]*10,10)
    assert constant['mcse']==0 and constant['empirical_outcome_correlation'] is None
    assert constant['marginal_ci95'][1]>0  # zero events do not establish zero risk


def test_all_removed_artifacts_reconstruct_original_bytes():
    manifest=json.loads(artifacts.MANIFEST.read_text())
    assert manifest['files']==112 and manifest['removed']==72
    assert artifacts.verify()==72
    for entry in manifest['entries']:
        if entry['action']=='removed_losslessly':
            assert hashlib.sha256(artifacts.reconstruct(entry)).hexdigest()==entry['sha256']


def test_saved_control_pairs_and_native_gates_without_refitting(tmp_path):
    import shutil
    from pathlib import Path
    from validation.ordinal_dif_paired import control
    source=control.HERE/'evidence_control'
    old_summary=(control.OLD/'summary.json').read_bytes()
    for path in source.glob('rep_*.json'):shutil.copyfile(path,tmp_path/path.name)
    for path in source.glob('data_*.csv'):shutil.copyfile(path,tmp_path/path.name)
    result=control.analyze(tmp_path)
    assert result['crn_verified_datasets']==10
    assert (control.OLD/'summary.json').read_bytes()==old_summary
    assert all(r['valid']==10 and r['failed']==0 for r in result['control_summary']['methods'].values())
    for row in result['impact0_minus_impact_minus0_5'].values():
        assert row['holm_event']['complete_pairs']==10
        assert row['holm_event']['mean']==0 and row['holm_event']['marginal_ci95'][1]>0
    for path in source.glob('rep_*.json'):
        r=json.loads(path.read_text())
        assert r['crn']['identical_reference_rows'] and r['crn']['focal_monotone']
        for hypotheses in r['methods']['GRM_LR']['fits'].values():
            for f in hypotheses.values():
                assert f['converged'] and f['quadrature_ok'] and f['information_available'] and not f['at_bound']
                for anchor in r['anchors']:
                    j=r['items'].index(anchor)
                    assert f['a'][0][j]==f['a'][1][j] and f['b'][0][j]==f['b'][1][j]
