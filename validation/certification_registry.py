"""Scoped evidence inventory and independent, no-fit multiplicity comparisons.

Run from repository root: python -m validation.certification_registry --write.
This never fits a model, changes an inference gate or upgrades archived evidence.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
from statsmodels.stats.multitest import multipletests

from core.dif_classical import adjust_pvalues
from core.example_datasets import analysis_catalog

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'validation/certification'
BASE = 'd9a62666ec6bdc879b6cac9c391cd5bbcb8ae480'
# Reuse the independently predeclared PR #139 tolerance; not fitted to new results.
ATOL = 1e-10
STATUSES = {'CERTIFIED', 'PARTIALLY CERTIFIED', 'NOT CERTIFIED', 'UNSUPPORTED'}
PHASE1 = 'validation/r_mirt_phase1/final_summary.json'
LORD = 'validation/lord_rasch_free_scale/results/wide/report.json'
FWER = 'validation/grm_fwer_sequential/results/summary.json'


def read(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def multiplicity_evidence():
    """Finite valid-p domain; no claim about the calibration of input p-values."""
    cases = [[], [np.nan], [0.], [1.], [0., 1., np.nan],
             [.001, .02, .02, .04, .2, .5, .9, np.nan, 1e-8, .049],
             [0., 0., 1., 1., .05, .05, .05]]
    rng = np.random.default_rng(20261105)
    for n in (2, 17, 101):
        p = rng.uniform(size=n)
        cases.extend([p.tolist(), p[::-1].tolist()])
    results = {}
    for method, ref in [('BH', 'fdr_bh'), ('BY', 'fdr_by'),
                        ('Holm', 'holm'), ('Bonferroni', 'bonferroni')]:
        delta = 0.
        for case in cases:
            p = np.asarray(case, float)
            ok = np.isfinite(p)
            actual = adjust_pvalues(p, method)
            assert np.isnan(actual[~ok]).all()
            if ok.any():
                expected = multipletests(p[ok], method=ref)[1]
                error = float(np.max(np.abs(actual[ok] - expected)))
                assert error <= ATOL, (method, error)
                delta = max(delta, error)
        results[method] = {'max_absolute_difference': delta,
                           'cases': len(cases), 'atol': ATOL, 'rtol': 0}
    return {
        'scope': 'Adjusted p-values only, finite inputs in [0,1], NaNs excluded/preserved; '
                 'recorded fixtures only. No Type-I, FWER or FDR certification of any DIF method.',
        'reference': 'statsmodels.stats.multitest.multipletests',
        'seed': 20261105,
        'inputs': [[None if np.isnan(v) else v for v in case] for case in cases],
        'versions': {'python': platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ('numpy', 'statsmodels')}},
        'source_sha256': {p: digest(p) for p in
                          ('core/dif_classical.py', 'validation/certification_registry.py')},
        'results': results,
    }


def cells(measured):
    rows = []

    def add(model, output, estimator, reference, evidence, status, limitation, tolerance=None):
        rows.append(dict(model=model, output=output, estimator=estimator, reference=reference,
                         tolerance=tolerance, evidence=evidence,
                         CI='Required check for this revision; consult PR checks (not scientific evidence)',
                         status=status, limitation=limitation))

    # Archived R calibrations certify neither a model nor sampling inference.
    phase = read(PHASE1)
    for model in ('Rasch', '1PL', '2PL', '3PL', '4PL'):
        r = phase['models'][model]
        label = {'Rasch': 'native Rasch (free SD)', '1PL': 'library fixed-a=1 1PL'}.get(model, model)
        for output, field in [('parameters', 'max_absolute_parameter_difference'),
                              ('category probabilities', 'max_absolute_ICC_difference'),
                              ('test information', 'max_absolute_test_information_difference'),
                              ('logLik', 'max_absolute_loglik_difference')]:
            add(label, output, 'ML', 'R mirt 1.48', [PHASE1],
                'NOT CERTIFIED' if model == '4PL' else 'PARTIALLY CERTIFIED',
                f"Archived fixture only; max absolute difference={r[field]:.12g}; "
                f"{r['likelihood_comparison']}; R converged={r['R_converged']}. "
                'No independently justified general equivalence tolerance or recovery certificate.')

    models = [
        ('native Rasch (free SD)', 'ML'), ('library fixed-a=1 1PL', 'ML'),
        ('common-a 1PL', 'ML'), ('2PL', 'ML'), ('3PL', 'ML'), ('3PL', 'MAP'),
        ('4PL', 'ML'), ('4PL', 'MAP'), ('GRM', 'ML'), ('PCM', 'ML'),
        ('GPCM', 'ML'), ('RSM', 'ML'), ('NRM', 'ML'), ('MIRT', 'ML'),
        ('MIRT', 'MAP'), ('bifactor', 'ML'), ('bifactor', 'MAP')]
    for model, estimator in models:
        for output in ('SE', 'CI coverage', 'parameter recovery', 'person scores and posterior SD'):
            evidence = ['validation/SCIENTIFIC_READINESS_2026-10-31.md']
            limitation = 'Existing scoped evidence does not establish all mandatory checks for this output.'
            if model in ('3PL', '4PL'):
                evidence = ['validation/evidence/p7_binary34_se/coverage.json']
                limitation = ('Historical coverage decisions are estimator/parameter/design specific; '
                              '4PL ML Wald intervals remain suppressed by policy. MAP is not ML.')
            if model in ('MIRT', 'bifactor'):
                limitation = ('Identification/structure/scale specific; ML references do not validate MAP, '
                              'bifactor recovery or inferential outputs. Preserve disabled inference.')
            if output == 'person scores and posterior SD':
                evidence = ['tests/test_independent_references.py']
                limitation = ('Independent same-grid EAP check covers the 2PL fixture only; '
                              'no cross-model, MAP, tail/grid or frequentist-coverage extrapolation.')
            add(model, output, estimator, 'Archived study or targeted test; scope must be reviewed', evidence,
                'PARTIALLY CERTIFIED' if model == '2PL' and output == 'person scores and posterior SD'
                else 'NOT CERTIFIED', limitation)

    for model in ('GRM', 'PCM', 'GPCM', 'NRM'):
        for output in ('category probabilities', 'item/test information and conditional SEM'):
            add(model, output, 'fixed coefficients', 'Independent analytic equations/finite differences',
                ['tests/test_scientific_gap_closure.py'], 'PARTIALLY CERTIFIED',
                'Formula checks conditional on supplied coefficients, not independent calibration, '
                'parameter uncertainty, category recovery or universal validity.')

    for output in ('chi-square and p', 'contrast covariance', 'latent SD and parameters'):
        add('1PL Lord free-SD Rasch', output, 'separate marginal ML; a=1, mean=0, SD free',
            'R mirt 1.48 Rasch / Oakes; native 61/101 and common GH101', [LORD],
            'PARTIALLY CERTIFIED',
            'Actual R executed; same 741 complete-case cohort. Max relative chi-square 1.671e-5. '
            'Full nuisance covariance and anchor delta method; no general DIF-FWER, power or CI coverage.',
            {'relative_chi_square': .001} if output == 'chi-square and p' else None)
    add('1PL Lord legacy fixed SD', 'inferential p/q', 'separate ML; SD=1', 'No valid free-scale reference',
        ['core/dif_lord_policy.py'], 'UNSUPPORTED', 'Old fixed-SD path remains descriptive; never promote via Wald SD diagnostics.')
    for model in ('1PL Wald', '2PL Wald', '2PL Lord', 'SIBTEST', 'Raju', 'selected-anchor DIF'):
        add(model, 'Type-I/FWER', 'method-specific', 'No general familywise validation',
            ['validation/DIF_CENTER.md'], 'NOT CERTIFIED', 'Numerical agreement is not operating-characteristic evidence.')
    add('fixed-anchor GRM-LR', 'Holm FWER', 'GH101 optimization / GH161 audit',
        'Frozen independent raw audit; sequential exact binomial intervals', [FWER], 'NOT CERTIFIED',
        '852 completed pairs; only registered look500 reached: 29/500 and 25/500 events. '
        'INCONCLUSIVE; 1000/1500 looks not executed. Prefix852 is descriptive, not a new look.')
    for model in ('3PL bootstrap DIF', '4PL bootstrap DIF'):
        add(model, 'inferential p', 'parametric bootstrap', 'Restricted simulation evidence',
            ['core/irt_dif34_bootstrap.py'], 'NOT CERTIFIED',
            'No general eligibility from model name; preserve existing fail-closed UI/export gates.')
    for model in ('Mantel ordinal', 'ordinal LR total', 'ordinal LR rest'):
        add(model, 'operational Holm FWER', 'observed-score matching', 'Registered latent-null GRM experiments',
            ['validation/DIF_VALIDATION.md'], 'NOT CERTIFIED',
            'Impact/design-specific error evidence; observed-score and latent nulls differ. '
            'Numerical p-adjustment correctness cannot establish valid input p-values.')

    for model, output, evidence, limitation in [
        ('NRM', 'Wright map', 'core/irt_b14_diagnostics.py', 'Nominal labels are not ordered thresholds; retain disabled map.'),
        ('NRM', 'ordinal S-X2 / infit/outfit', 'core/irt_b14_diagnostics.py', 'No justified NRM fit equivalent; do not fabricate.'),
        ('G theory', 'fixed facets / unbalanced designs', 'core/generalizability.py', 'Current API rejects these unsupported designs.')]:
        add(model, output, 'not available', 'Method/API restriction', [evidence], 'UNSUPPORTED', limitation)
    for model, reference, evidence in [
        ('CTT reliability', 'pingouin alpha; psych archived coefficients', 'tests/test_independent_references.py'),
        ('G theory balanced p x i', 'Independent ANOVA sums of squares', 'tests/test_scientific_gap_closure.py'),
        ('dimensionality / CFA', 'Existing audit; no complete per-output independent certificate', 'validation/SCIENTIFIC_AUDIT_PHASE2.md'),
        ('missing data', 'Observed-value and mask invariants', 'validation/SCIENTIFIC_AUDIT_PHASE2.md')]:
        add(model, 'supported numerical outputs', 'route specific', reference, [evidence],
            'PARTIALLY CERTIFIED', 'One-fixture/software/formula checks do not certify every output or uncertainty.')
    for method, result in measured['results'].items():
        add('multiplicity '+method, 'adjusted p-values (recorded valid-p fixtures only)', 'closed form',
            measured['reference']+' '+measured['versions']['statsmodels'],
            ['validation/certification/multiplicity.json', 'tests/test_certification_registry.py'],
            'CERTIFIED', measured['scope'], {'atol': result['atol'], 'rtol': 0})
    for route in analysis_catalog():
        add(route, 'screen / Word / CSV / XLSX / figures', 'actual route result',
            'Existing AppTests and export contracts; no independent scientific certificate',
            ['tests/test_word_standard.py', 'tests/test_result_panels.py'], 'PARTIALLY CERTIFIED',
            'Catalog coverage, not a claim that every route supports every format. Software consistency '
            'does not certify statistics. Generated Python code remains UNVERIFIED unless executed.')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    measured = multiplicity_evidence()
    rows = cells(measured)
    counts = {status: sum(r['status'] == status for r in rows) for status in sorted(STATUSES)}
    if args.write:
        OUT.mkdir(exist_ok=True)
        (OUT / 'multiplicity.json').write_text(json.dumps(measured, ensure_ascii=False, indent=2)+'\n')
        (OUT / 'matrix.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
        paths = {p for r in rows for p in r['evidence']}
        manifest = {'base': BASE, 'scope': 'Conservative claim inventory, not an exhaustive output certificate',
                    'counts': counts, 'routes': len(analysis_catalog()),
                    'sha256': {p: digest(p) for p in sorted(paths)}}
        (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'cells': len(rows), 'counts': counts, 'new_fits': 0, 'R_execution_this_run': False}))


if __name__ == '__main__':
    main()
