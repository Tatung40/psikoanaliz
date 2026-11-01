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
    assert manifest['files']==112 and manifest['removed']==62
    assert artifacts.verify()==62
    for entry in manifest['entries']:
        if entry['action']=='removed_losslessly':
            assert hashlib.sha256(artifacts.reconstruct(entry)).hexdigest()==entry['sha256']
