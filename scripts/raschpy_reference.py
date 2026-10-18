"""RaschPy SLM reference values for the B8 Rasch module (second, independent reference).

RaschPy estimates item locations by PAIR (pairwise/Bradley-Terry), zero-centred over
items; that is NOT marginal maximum likelihood, so its item locations are compared
with core.rasch only after centring, as agreement indicators, never as equality.
Formula-level checks use RaschPy's own estimates as inputs (WLE given RaschPy items;
Infit/Outfit given RaschPy items and persons).

A fresh SLM object is built for every calibration. Run from the repository root:
    python scripts/raschpy_reference.py
Writes tests/fixtures/rasch/raschpy_results.csv and raschpy_session.txt.
"""
from importlib.metadata import version
from pathlib import Path
import platform
import warnings
import numpy as np
import pandas as pd
from raschpy import SLM

FIX = Path("tests/fixtures/rasch")


def main():
    rows = []
    def add(case, quantity, row, col, value):
        rows.append({"case": case, "quantity": quantity, "row": str(row), "col": str(col), "value": value})
    for case in pd.read_csv(FIX / "cases.csv").case:
        data = pd.read_csv(FIX / f"data_{case}.csv", na_values="NA")
        data.index = [str(i + 1) for i in range(len(data))]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = SLM(data)                                   # fresh object per calibration
            model.calibrate()                                   # PAIR, default settings
            model.std_errors(se_method="sandwich")
            model.person_estimates()                            # Warm WLE
            model.fit_statistics()
        for w in caught:
            add(case, "warning", type(w.message).__name__, "", np.nan); rows[-1]["row"] = str(w.message)[:200]
        add(case, "constant", "", "", float(model.constant))
        add(case, "n_persons_used", "", "", int(model.no_of_persons))
        add(case, "n_persons_fit", "", "", int(len(model.person_infit_ms)))
        # RaschPy 1.4.0 docstrings name these `diffs` and `person_locations`; the attributes
        # actually set are `items` (Series after calibrate) and `persons` (verified at run time).
        for item, v in model.items.items():
            add(case, "item_location", item, "", float(v))
        for item, v in model.item_se.items():
            add(case, "item_se_sandwich", item, "", float(v))
        for person, v in model.persons.items():
            add(case, "person_wle", person, "", float(v))
        for name, series in (("Infit MNSQ", model.item_infit_ms), ("Infit ZSTD", model.item_infit_zstd),
                             ("Outfit MNSQ", model.item_outfit_ms), ("Outfit ZSTD", model.item_outfit_zstd)):
            for item, v in series.items():
                add(case, "item_fit", item, name, float(v))
        for name, series in (("Infit MNSQ", model.person_infit_ms), ("Infit ZSTD", model.person_infit_zstd),
                             ("Outfit MNSQ", model.person_outfit_ms), ("Outfit ZSTD", model.person_outfit_zstd)):
            for person, v in series.items():
                add(case, "person_fit", person, name, float(v))
        for name in ("isi", "item_reliability", "psi", "person_reliability"):
            add(case, "test_stat", name, "", float(getattr(model, name)))
    out = pd.DataFrame(rows)
    out.to_csv(FIX / "raschpy_results.csv", index=False, float_format="%.12g")
    (FIX / "raschpy_session.txt").write_text(
        f"RaschPy {version('raschpy')} (PyPI wheel raschpy-1.4.0-py3-none-any.whl)\n"
        f"Python {platform.python_version()} | numpy {np.__version__} | pandas {pd.__version__}\n"
        "SLM(responses) default extreme_persons=True; calibrate() defaults (PAIR, method='log-lik',\n"
        "data-adaptive constant); std_errors(se_method='sandwich'); person_estimates() (Warm WLE);\n"
        "fit_statistics() defaults.\n", encoding="utf-8")
    print("written", len(out), "rows")


if __name__ == "__main__":
    main()
