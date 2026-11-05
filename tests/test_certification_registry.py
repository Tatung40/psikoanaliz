"""Frozen scientific claims, independent closed forms; no fits or simulations."""
import json
from collections import Counter

import pytest

from validation.certification_registry import (
    ROOT, OUT, STATUSES, PHASE1, LORD, FWER, cells, digest, multiplicity_evidence, read,
)


def test_scoped_registry_is_reproducible_complete_and_evidence_is_immutable():
    saved = [json.loads(line) for line in (OUT/'matrix.jsonl').read_text().splitlines()]
    measured = read('validation/certification/multiplicity.json')
    assert saved == cells(measured)
    manifest = read('validation/certification/manifest.json')
    assert dict(Counter(r['status'] for r in saved)) == manifest['counts']
    required = {'model', 'output', 'estimator', 'reference', 'tolerance', 'evidence', 'CI', 'status', 'limitation'}
    assert len({(r['model'], r['output'], r['estimator']) for r in saved}) == len(saved)
    for row in saved:
        assert set(row) == required and row['status'] in STATUSES
        assert row['reference'] and row['limitation'] and row['evidence']
        for path in row['evidence']:
            assert (ROOT/path).is_file(), path
    for path, expected in manifest['sha256'].items():
        assert digest(path) == expected, path
    from core.example_datasets import analysis_catalog
    actual = {r['model'] for r in saved if r['output'].startswith('screen / Word')}
    assert actual == set(analysis_catalog())
    assert len(actual) == manifest['routes']


@pytest.mark.parametrize('method', ['BH', 'BY', 'Holm', 'Bonferroni'])
def test_certified_multiplicity_has_actual_independent_reference_and_scoped_limits(method):
    measured = multiplicity_evidence()
    saved = read('validation/certification/multiplicity.json')
    # Versions can differ between runners; numerical tolerance cannot.
    assert measured['inputs'] == saved['inputs']
    result = measured['results'][method]
    assert result['atol'] == 1e-10 and result['rtol'] == 0
    assert result['max_absolute_difference'] <= result['atol']
    for path, expected in saved['source_sha256'].items():
        assert digest(path) == expected
    certified = [r for r in cells(saved) if r['status'] == 'CERTIFIED']
    assert len(certified) == 4
    assert all(r['output'].startswith('adjusted p-values') and 'No Type-I' in r['limitation']
               for r in certified)


def test_nonconverged_4pl_and_unreached_fwer_looks_cannot_be_promoted():
    phase = read(PHASE1)['models']['4PL']
    assert phase['R_converged'] is False and phase['scientific_pass'] is False
    assert phase['max_absolute_loglik_difference'] == pytest.approx(.0002288126379426103)
    fwer = read(FWER)
    assert fwer['scientific_status'] == 'INCONCLUSIVE / NOT CERTIFIED'
    assert fwer['unreached_looks'] == [1000, 1500]
    rows = cells(read('validation/certification/multiplicity.json'))
    assert all(r['status'] != 'CERTIFIED' for r in rows
               if r['model'] in ('4PL', 'fixed-anchor GRM-LR'))


def test_free_sd_lord_retains_actual_r_evidence_without_claiming_fwer_certification():
    report = read(LORD)
    assert report['R_execution'] == 'EXECUTED'
    for path, expected in report['production_source_sha256'].items():
        assert digest(path) == expected, path
    for points in ('61', '101'):
        r = report['comparisons'][points]
        assert r['Statistic']['max_relative'] <= report['limits']['relative_chi_square'] == .001
        for group in ('R', 'F'):
            assert r[group]['Python_converged'] is True
            assert r[group]['R_diagnostics']['converged'] is True
            assert r[group]['R_diagnostics']['identification'] == 'a=1; mean=0; SD free'
    rows = [r for r in cells(read('validation/certification/multiplicity.json'))
            if r['model'] == '1PL Lord free-SD Rasch']
    assert len(rows) == 3 and all(r['status'] == 'PARTIALLY CERTIFIED' for r in rows)
