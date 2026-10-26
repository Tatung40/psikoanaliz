"""Read existing evidence only: prevent documentation from certifying 4PL."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'validation/r_mirt_phase1'


def test_final_classification_preserves_nonconverged_uncertified_4pl():
    summary=json.loads((ROOT/'final_summary.json').read_text())
    evidence=json.loads((ROOT/'observed_information/report.json').read_text())
    four=summary['models']['4PL']
    assert four['status']=='PARTIAL NUMERICAL AGREEMENT / NOT CONVERGED / NOT CERTIFIED'
    assert four['R_converged'] is evidence['original_convergence']['converged'] is False
    assert four['independent_estimation'] is False  # Python-seeded R diagnostic.
    assert all(m['scientific_pass'] is False for m in summary['models'].values())
    assert evidence['convergence_trace']['exact_last_pair_satisfied'] is False
    assert evidence['convergence_trace']['parameters_not_below_TOL']==29
    assert four['status'] in (ROOT/'README.md').read_text()


def test_hessian_summary_copies_preserved_results_without_recalculation():
    final=json.loads((ROOT/'final_summary.json').read_text())['four_PL_information']
    evidence=json.loads((ROOT/final['evidence']).read_text())
    assert final['min_active_face_eigenvalue']==evidence['active_face_information']['fine_eigenvalues'][0]
    assert final['scaled_condition_number']==evidence['diagonally_scaled_active_face']['condition_number']
    assert final['M12_weakest_squared_loading']==evidence['active_face_modes'][0]['M12_squared_loading']
    assert final['new_fits']==evidence['new_fits']==0
