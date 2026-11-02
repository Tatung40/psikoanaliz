# jMetrik/psychometrics gap closure — 2026-11-02

Clean-room implementations from published formulas; validated only against published
examples (no simulation, no new calibration except one LSAT7 2PL fit). Not wired into the
UI yet. Status for every item: **IMPLEMENTED / PUBLISHED-EXAMPLE MATCHED / NOT CERTIFIED**.

| Feature | Module | Reference | Tolerance | Max observed difference |
|---|---|---|---|---|
| Stocking–Lord (Q1Q2) | `core/irt_linking.py` | STUIRT example 1 | 1e-6 | 1.6e-7 |
| Haebara (Q1Q2 / Q1) | `core/irt_linking.py` | STUIRT example 1 | 1e-4 | 1.5e-6 / 3.0e-5 |
| Mean/sigma, mean/mean | `core/irt_linking.py` | exact known transform | 1e-12 | ≈0 |
| IRT true-score equating | `core/irt_equating.py` | POLYEQUATE, 36 items | 1e-5 | 1.7e-6 (θ 6.5e-7) |
| IRT observed-score equating | `core/irt_equating.py` | Equating Recipes, 36 items | 1e-5 | 4.9e-6 |
| Guttman λ1/λ2/λ3/λ5/λ6 | `core/reliability_extended.py` | R psych, Feldt–Brennan matrix | 1e-6 | ≈4e-8 |
| Feldt–Gilmer, Feldt–Brennan | `core/reliability_extended.py` | compound-symmetry identity (= α) | 1e-12 | ≈0 |
| GPA quartimin/oblimin/geomin | `core/factor_rotation.py` | R GPArotation | 1e-4 | 4.9e-9 |
| Python mirt 2PL S-X² | `core/irt_library_diagnostics.py` label | R mirt, LSAT7 | 1e-2 | 2.2e-3 |

Limits: dichotomous 3PL-family linking/equating only (no GPCM/GRM mixed format); equating
standard errors not implemented; Feldt–Gilmer/Brennan have no external numeric reference;
S-X² agreement is one public data set (2PL only — 1PL/3PL/4PL labels unchanged); no
Ramsay NIRT. These results are not FWER/coverage certification and do not alter any
existing gate, label policy or archived evidence. Tests: `tests/test_psychometrics_reference.py`.
