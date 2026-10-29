# Independent R mirt — ML 3PL/4PL with observed-information SEs

Gap: the 3PL R comparison (`validation/r_mirt_phase1`) had no SE comparison and the 4PL R fit never converged.
Design and criteria: docstring of `compare.py` (fixed before any R run). Fixture: N = 5000, 15 items, truth of
`validation/irt_3pl4pl_root_cause.py`; the first seed in 1..50 whose Python ML fit has no parameter on a box
bound (Python fixes bound parameters, R's bounds are open, so only interior solutions are the same problem).
Result of `select_seeds()`: 3PL seed 13; **4PL: none of 1..50** — every 4PL ML fit had a parameter on a bound,
so the 4PL comparison is reported NOT VERIFIED (no seed search beyond the pre-set limit). R runs only in
`.github/workflows/r-mirt-34.yml`.
