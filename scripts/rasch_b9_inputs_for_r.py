"""Python-side inputs consumed by scripts/mirt_rasch_b9_reference.R (B9).

Q3: B8 MML difficulties, latent SD and WLE scores for ld_pair and rasch_missing, so that
mirt computes Q3 from exactly the same residuals. DIF: the final purified anchor sets
from core.rasch_dif for dif_known and dif_none (mirt then fits multipleGroup with those
anchors; purification itself is our algorithm and is not part of the comparison).
Run from the repository root before the R script.
"""
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.rasch import rasch_analysis
from core.rasch_dif import dif_analysis

F8, F9 = Path("tests/fixtures/rasch"), Path("tests/fixtures/rasch_b9")


def main():
    for case, folder in (("ld_pair", F9), ("rasch_missing", F8)):
        a = rasch_analysis(pd.read_csv(folder / f"data_{case}.csv", na_values="NA"))
        pd.DataFrame({"Madde": a["items"]["Madde"], "b": a["fit"]["b"]}).to_csv(
            F9 / f"py_q3_items_{case}.csv", index=False, float_format="%.17g")
        pd.DataFrame({"theta": a["persons"]["WLE θ"], "sd": a["fit"]["sd"]}).to_csv(
            F9 / f"py_q3_persons_{case}.csv", index=False, float_format="%.17g")
    for case in ("dif_known", "dif_none"):
        d = pd.read_csv(F9 / f"data_{case}.csv", na_values="NA")
        out = dif_analysis(d.drop(columns="Grup"), d["Grup"], "R", "F")
        pd.DataFrame({"anchor": out["anchors"]}).to_csv(F9 / f"py_dif_anchors_{case}.csv", index=False)


if __name__ == "__main__":
    main()
